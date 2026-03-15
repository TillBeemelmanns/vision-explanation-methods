import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T

from ml_wrappers.model.image_model_wrapper import PytorchDRiseWrapper
from vision_explanation_methods.explanations import drise

def save_pure_image(data, filename, cmap=None, is_mask=False):
    """Save an image without any axes, borders or whitespace."""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(data.shape[1]/100, data.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    if is_mask:
        # Saliency map is usually (C, H, W), take mean over channels and normalize
        if len(data.shape) == 3:
            data = np.mean(data, axis=0)
        ax.imshow(data, aspect='auto', cmap=cmap, interpolation='bilinear')
    else:
        # Original image is (H, W, C)
        ax.imshow(data, aspect='auto')
        
    plt.savefig(filename, dpi=100)
    plt.close(fig)

# COCO 91-class label names (index 0 = __background__)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def smooth_saliency(sal_map, kernel_size=21, sigma=5.0):
    """Apply Gaussian smoothing to a 2D saliency map using torchvision."""
    blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tensor = torch.from_numpy(sal_map).float().unsqueeze(0).unsqueeze(0)
    smoothed = blur(tensor)
    return smoothed.squeeze().numpy()


def print_detections(detections):
    """Print all detections in a formatted table and return the list of labels."""
    det = detections[0]
    n = det.bounding_boxes.shape[0]
    print(f"\n{'Idx':>4}  {'Class':>4}  {'Label':<20s}  {'Score':>7s}  {'BBox (x1,y1,x2,y2)'}")
    print("-" * 75)
    labels = []
    for i in range(n):
        class_id = torch.argmax(det.class_scores[i]).item()
        score = det.class_scores[i][class_id].item()
        bbox = det.bounding_boxes[i].cpu().numpy()
        name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        labels.append((class_id, name))
        print(f"{i:>4d}  {class_id:>4d}  {name:<20s}  {score:>7.3f}  "
              f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
    print()
    return labels


def generate_custom_visualizations(image_path, num_masks=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enable cudnn auto-tuner for faster convolutions
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Load Image
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)

    # Save original image
    img_np = np.array(img_pil)
    save_pure_image(img_np, "output_original.jpg")
    print("Saved: output_original.jpg")

    # 2. Load Pretrained Model
    unwrapped_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT'
    ).to(device)
    unwrapped_model.eval()

    # Wrap for D-RISE
    num_classes = 91
    model = PytorchDRiseWrapper(unwrapped_model, num_classes)

    # 3. Get Base Detections (fp16 for speed on GPU)
    print("Running initial detection...")
    use_fp16 = device.type == 'cuda'
    with torch.no_grad():
        if use_fp16:
            with torch.amp.autocast('cuda'):
                detections = model.predict(img_tensor)
        else:
            detections = model.predict(img_tensor)

    if detections[0].bounding_boxes.shape[0] == 0:
        print("No detections found in the image.")
        return

    # Print all detections and let the user pick BEFORE running D-RISE
    det_labels = print_detections(detections)
    n_dets = len(det_labels)

    while True:
        choice = input(f"Select detection index to visualize [0-{n_dets - 1}]: ").strip()
        if choice.isdigit() and 0 <= int(choice) < n_dets:
            best_det_idx = int(choice)
            break
        print(f"Invalid input. Please enter a number between 0 and {n_dets - 1}.")

    class_id, class_name = det_labels[best_det_idx]
    print(f"\nExplaining detection {best_det_idx}: {class_name} (class {class_id})")

    # Subset target_detections to only the selected detection.
    # This makes D-RISE compute saliency for just 1 detection instead of all,
    # drastically reducing computation and memory.
    selected_detections = [detections[0].get_by_index([best_det_idx])]

    # 4. Run D-RISE saliency (memory-efficient online accumulation + fp16)
    print(f"Generating D-RISE saliency maps with {num_masks} masks...")
    saliency_results = drise.DRISE_saliency(
        model=model,
        image_tensor=img_tensor,
        target_detections=selected_detections,
        number_of_masks=num_masks,
        mask_res=(12, 12),
        device=device,
        verbose=True,
        use_fp16=use_fp16,
    )

    # We only have one image in batch, one detection selected
    img_saliency = saliency_results[0]

    if not img_saliency:
        print("D-RISE produced no saliency map for this detection.")
        return

    # Free model from GPU now that we're done with inference
    del unwrapped_model, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Get the saliency map tensor (C, H, W) — index 0 since we selected one detection
    sal_map = img_saliency[0]['detection'].cpu().numpy()
    # Average across channels if needed (D-RISE usually produces 3 identical channels)
    sal_map_single = np.mean(sal_map, axis=0)

    # Gaussian smoothing for nicer visualization
    sal_map_single = smooth_saliency(sal_map_single, kernel_size=21, sigma=5.0)

    # 5. Save Saliency Map Only (Blue to Red - 'jet' or 'RdYlBu_r')
    save_pure_image(sal_map_single, "output_saliency_only.jpg", cmap='jet', is_mask=True)
    print("Saved: output_saliency_only.jpg (Pure Saliency)")

    # Save Overlay
    fig = plt.figure(frameon=False)
    fig.set_size_inches(img_np.shape[1]/100, img_np.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img_np, aspect='auto')
    ax.imshow(sal_map_single, aspect='auto', cmap='jet', alpha=0.50, interpolation='bilinear')

    plt.savefig("output_overlay.jpg", dpi=100)
    plt.close(fig)
    print("Saved: output_overlay.jpg (Overlay)")

    # 7. Save Overlay with Bounding Box
    bbox = detections[0].bounding_boxes[best_det_idx].cpu().numpy()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(img_np.shape[1]/100, img_np.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img_np, aspect='auto')
    ax.imshow(sal_map_single, aspect='auto', cmap='jet', alpha=0.5, interpolation='bilinear')

    # Add green bounding box
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    rect = plt.Rectangle((xmin, ymin), width, height, linewidth=6, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    plt.savefig("output_overlay_bbox.jpg", dpi=100)
    plt.close(fig)
    print("Saved: output_overlay_bbox.jpg (Overlay with Bounding Box)")

    # 8. Save Original Image with Bounding Box only (no saliency)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(img_np.shape[1]/100, img_np.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img_np, aspect='auto')

    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    rect = plt.Rectangle((xmin, ymin), width, height, linewidth=6, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    plt.savefig("output_detection.jpg", dpi=100)
    plt.close(fig)
    print("Saved: output_detection.jpg (Original + Bounding Box)")
    plt.close(fig)
    print("Saved: output_overlay_bbox.jpg (Overlay with Bounding Box)")

if __name__ == "__main__":
    image_path = "/home/beemelmanns/Documents/github/vision-explanation-methods/python/vision_explanation_methods/images/pedestrian.jpg"
    generate_custom_visualizations(image_path)
