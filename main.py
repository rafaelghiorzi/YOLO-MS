from yolov8.test.test_model import analyze_complete_model

# Example usage
if __name__ == "__main__":
    # Test with different model versions
    versions_to_test = ['n']  # You can add 's', 'm', 'l', 'x'
    
    for version in versions_to_test:
        print(f"\n{'='*60}")
        print(f"Testing YOLOv8-{version.upper()}")
        print(f"{'='*60}")
        
        try:
            outputs = analyze_complete_model("yolov8/test/sample.png", version=version)
            print(f"\n✅ YOLOv8-{version.upper()} analysis complete!")
        except FileNotFoundError:
            print("❌ Image file not found. Please ensure 'sample.png' exists in the yolov8 folder.")
            print("💡 Tip: You can use any .jpg, .png, or .jpeg image file.")
            break
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            break