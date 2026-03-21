import gradio as gr
import json
from PIL import Image, ImageDraw
import os
import argparse
import random

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize camera images and captions from a JSON file.")
    parser.add_argument(
        '--json-path', 
        type=str, 
        default='/data/runhui/captions/nuscenes_train_28130_caption_20250629.json',
        help='Path to the JSON file containing the data.'
    )
    parser.add_argument(
        '--camera-root', 
        type=str, 
        default='/data/zliu/', 
        help='Root directory for the relative camera image paths.'
    )
    return parser.parse_args()

def load_data_from_path(json_path):
    """Loads data from a JSON file path."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out items that are empty or don't have camera_paths
    data = [dict(camera_paths={k.split('/')[1]:k for k in item['image']}, caption=item['predictions']) for item in data]
    
    if not data:
        raise ValueError("No valid items with 'camera_paths' found in the JSON file.")
    return data

def stitch_images(images):
    """Stitches 6 images into a 2x3 grid."""
    assert len(images) == 6, f"Expected 6 images, got {len(images)}"

    # Define a fixed size for each grid cell
    img_width, img_height = 400, 225
    
    # Resize all images to the fixed size
    resized_images = [img.resize((img_width, img_height), Image.Resampling.LANCZOS) for img in images]

    # Create the final grid image
    grid_image = Image.new('RGB', (img_width * 3, img_height * 2))

    # Paste the top row (front cameras)
    for i, img in enumerate(resized_images[:3]):
        grid_image.paste(img, (i * img_width, 0))

    # Paste the bottom row (back cameras)
    for i, img in enumerate(resized_images[3:]):
        grid_image.paste(img, (i * img_width, img_height))
        
    return grid_image

def get_item_data(index, data, camera_root):
    """
    Retrieves camera images, stitches them into a 2-row grid, and returns the result with caption.
    """
    index = int(index)
    item = data[index]
    
    caption = item['caption']
    camera_paths = item.get('camera_paths', {})
    
    # Define the desired order for a 2x3 grid view
    # Top row: Front Left, Front, Front Right
    # Bottom row: Back Left, Back, Back Right
    camera_order = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
    ]
    
    loaded_images = []
    info_text_parts = [f"Displaying item {index + 1} of {len(data)}"]

    for cam_name in camera_order:
        relative_path = camera_paths.get(cam_name).replace('samples/', 'trainval/samples/')
        
        full_path = os.path.join(camera_root, relative_path)
        info_text_parts.append(f"- {cam_name}: {full_path}")
        
        img = Image.open(full_path)
        loaded_images.append(img)

    # Stitch the loaded images together into a grid
    stitched_image = stitch_images(loaded_images)
    info_text = "\n".join(info_text_parts)
    
    return stitched_image, caption, info_text

def create_ui(initial_data, camera_root):
    """
    Creates the Gradio user interface.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📷 Camera Caption Visualizer")
        gr.Markdown("Click the button to view a new random sample from the dataset.")

        # Store data and config in state variables
        data_state = gr.State(initial_data)
        camera_root_state = gr.State(camera_root)

        with gr.Row():
            with gr.Column(scale=1):
                random_button = gr.Button("🔄 Show Random Sample", variant="primary")
                info_display = gr.Textbox(label="Image Paths", interactive=False, lines=8)
            
            with gr.Column(scale=3):
                image_display = gr.Image(label="Stitched Camera View", type="pil")
                caption_display = gr.Textbox(label="Generated Caption", lines=5, interactive=False)

        def show_random_sample(data, camera_root):
            if not data:
                placeholder = Image.new('RGB', (800, 225), color='grey')
                draw = ImageDraw.Draw(placeholder)
                draw.text((10, 10), "No data loaded.", fill='red')
                return placeholder, "No data loaded.", "No data loaded."
            
            random_index = random.randint(0, len(data) - 1)
            return get_item_data(random_index, data, camera_root)

        # Connect the button to the function
        random_button.click(
            show_random_sample,
            inputs=[data_state, camera_root_state],
            outputs=[image_display, caption_display, info_display]
        )
        
        # Trigger initial load for a random item when the UI is ready
        demo.load(
            show_random_sample,
            inputs=[data_state, camera_root_state],
            outputs=[image_display, caption_display, info_display]
        )
    
    return demo

if __name__ == "__main__":
    args = parse_args()
    data = load_data_from_path(args.json_path)
    # Pass the filename to the UI for display
    app = create_ui(data, args.camera_root)
    app.launch(server_name='0.0.0.0', debug=True)
