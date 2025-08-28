from pathlib import Path
from llama_index.core.llms import ImageBlock
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


def describe_image(
    multimodal_model: OpenAIMultiModal, image_path: Path, text_description: str
) -> str:
    """Describes an image using the multimodal model.

    Parameters
    ----------
    multimodal_model : LLM
        The multimodal model to use for image description.

    image_path : Path
        Path to the image.

    text_description : str
        Non AI generated supplementary text description to add to the image description.

    Returns
    -------
    str
        Description of the image.
    """
    prompt = "Describe the statistical figure in detail. Be specific. "
    "Focus only on insights you can infer from the figure, "
    "rather than color, size, or other non-data related aspects.\n\n"
    "Here is some additional contextual information about the figure:\n\n"
    f"'{text_description}'"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    try:
        str_description = multimodal_model.complete(
            prompt=prompt, image_documents=[ImageBlock(image=image_bytes)]
        ).text
    except Exception as e:
        print("Error occurred while describing image:", e)
        str_description = "Error occurred while describing image."
    return str_description
