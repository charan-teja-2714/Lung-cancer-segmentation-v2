import torch
import segmentation_models_pytorch as smp
# from torchsummary import summary

# Model configuration from training code
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=1,
    classes=4   # background + ADC + LCC + SCC
)

print("=" * 70)
print("MODEL ARCHITECTURE DETAILS")
print("=" * 70)

print("\n1. INPUT SIZE")
print("   - Input Shape: (1, 256, 256)")
print("   - Channels: 1 (Grayscale)")
print("   - Spatial Dimensions: 256 x 256")

print("\n2. ENCODER")
print("   - Architecture: EfficientNet-B4")
print("   - Pretrained: Yes (ImageNet)")
print("   - Input Channels: 1 (modified from 3)")

print("\n3. DECODER")
print("   - Architecture: U-Net++ (Nested U-Net)")
print("   - Skip Connections: Dense nested connections")
print("   - Deep Supervision: Available (not used in training)")

print("\n4. OUTPUT")
print("   - Output Channels: 4 classes")
print("   - Classes: [Background, ADC, LCC, SCC]")
print("   - Activation: None (logits)")
print("   - Final Activation: Softmax (applied via CrossEntropyLoss)")

print("\n5. ARCHITECTURE TYPE")
print("   - Type: 2D Segmentation")
print("   - Framework: segmentation_models_pytorch")

print("\n6. PARAMETER COUNT")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   - Total Parameters: {total_params:,}")
print(f"   - Trainable Parameters: {trainable_params:,}")
print(f"   - Model Size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")

print("\n7. DECODER FILTERS (U-Net++ Default)")
print("   - Decoder channels follow encoder feature maps")
print("   - EfficientNet-B4 encoder outputs:")
print("     * Stage 1: 24 channels")
print("     * Stage 2: 32 channels")
print("     * Stage 3: 56 channels")
print("     * Stage 4: 160 channels")
print("     * Stage 5: 448 channels (bottleneck)")

print("\n8. LOSS FUNCTION")
print("   - Combined Loss: CrossEntropyLoss + DiceLoss")
print("   - Weight: 1:1 ratio")

print("\n" + "=" * 70)

# Count parameters by layer type
print("\nPARAMETER BREAKDOWN:")
encoder_params = sum(p.numel() for name, p in model.named_parameters() if 'encoder' in name)
decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'decoder' in name)
segmentation_params = sum(p.numel() for name, p in model.named_parameters() if 'segmentation' in name)

print(f"   - Encoder Parameters: {encoder_params:,}")
print(f"   - Decoder Parameters: {decoder_params:,}")
print(f"   - Segmentation Head: {segmentation_params:,}")

print("\n" + "=" * 70)
