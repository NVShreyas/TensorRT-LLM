import asyncio
import base64
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

from tensorrt_llm.serve.openai_protocol import (
    ImageEditRequest,
    ImageGenerationRequest,
    VideoGenerationRequest,
)
from tensorrt_llm.visual_gen import VisualGen, VisualGenParams

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _detect_reference_media_type(
    data: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Literal["image", "video"]:
    """Classify an input_reference payload as image or video conditioning."""
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in _VIDEO_EXTENSIONS:
            return "video"
        if ext in _IMAGE_EXTENSIONS:
            return "image"

    if content_type:
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("image/"):
            return "image"

    if data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8\xff"):
        return "image"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"AVI ":
        return "video"

    # Preserve legacy behavior: unknown payloads are treated as images.
    return "image"


def _reference_file_extension(
    media_type: Literal["image", "video"],
    filename: Optional[str] = None,
) -> str:
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext:
            return ext
    return ".mp4" if media_type == "video" else ".png"


def _read_input_reference(
    input_reference,
) -> Tuple[bytes, Optional[str], Optional[str]]:
    if isinstance(input_reference, str):
        return base64.b64decode(input_reference), None, None

    filename = getattr(input_reference, "filename", None)
    content_type = getattr(input_reference, "content_type", None)
    data = input_reference.file.read()
    return data, filename, content_type


def _save_input_reference(
    input_reference,
    *,
    request_id: str,
    media_storage_path: str,
) -> Tuple[str, Literal["image", "video"]]:
    data, filename, content_type = _read_input_reference(input_reference)
    media_type = _detect_reference_media_type(data, filename=filename, content_type=content_type)
    ext = _reference_file_extension(media_type, filename)
    ref_path = os.path.join(media_storage_path, f"{request_id}_reference{ext}")
    with open(ref_path, "wb") as f:
        f.write(data)
    return ref_path, media_type


def parse_visual_gen_params(
    request: ImageGenerationRequest | VideoGenerationRequest | ImageEditRequest,
    id: str,
    generator: VisualGen,
    media_storage_path: Optional[str] = None,
) -> VisualGenParams:
    # Start from the pipeline's resolved defaults so unspecified request
    # fields keep the model's defaults instead of being overwritten with None.
    params = generator.default_params
    if params.extra_params is None:
        params.extra_params = {}

    if request.negative_prompt is not None:
        params.negative_prompt = request.negative_prompt
    if request.size is not None and request.size != "auto":
        params.width, params.height = map(int, request.size.split("x"))
    if request.guidance_scale is not None:
        params.guidance_scale = request.guidance_scale
    if request.guidance_rescale is not None:
        params.extra_params["guidance_rescale"] = request.guidance_rescale

    if isinstance(request, (ImageGenerationRequest, ImageEditRequest)):
        if request.num_inference_steps is not None:
            params.num_inference_steps = request.num_inference_steps
        elif isinstance(request, ImageGenerationRequest) and request.quality == "hd":
            params.num_inference_steps = 30
        if request.n is not None:
            params.num_images_per_prompt = request.n
        if isinstance(request, ImageEditRequest):
            if request.image is not None:
                if isinstance(request.image, list):
                    params.image = [base64.b64decode(image) for image in request.image]
                else:
                    params.image = [base64.b64decode(request.image)]
            if request.mask is not None:
                if isinstance(request.mask, list):
                    params.mask = [base64.b64decode(mask) for mask in request.mask]
                else:
                    params.mask = base64.b64decode(request.mask)

    elif isinstance(request, VideoGenerationRequest):
        if request.num_inference_steps is not None:
            params.num_inference_steps = request.num_inference_steps
        if request.n is not None:
            params.num_images_per_prompt = request.n

        if request.input_reference is not None:
            if media_storage_path is None:
                raise ValueError("media_storage_path is required when input_reference is provided")
            ref_path, media_type = _save_input_reference(
                request.input_reference,
                request_id=id,
                media_storage_path=media_storage_path,
            )
            if media_type == "video":
                params.video = ref_path
            else:
                params.image = ref_path

        params.frame_rate = request.fps
        params.num_frames = int(request.seconds * request.fps)

        if request.seed is not None:
            params.seed = int(request.seed)

    # Drop extra_params if we didn't end up with any — matches VisualGenParams
    # convention where None means "no extras" for pipelines that declare none.
    if not params.extra_params:
        params.extra_params = None

    return params


class AsyncDictStore:
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            self._items[key] = value

    async def update_fields(self, key: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            item.update(updates)
            return item

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.pop(key, None)

    async def list_values(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._items.values())


# Global stores shared by OpenAI entrypoints
# [request_id, dict]
VIDEO_STORE = AsyncDictStore()
