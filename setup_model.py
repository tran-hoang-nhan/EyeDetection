#!/usr/bin/env python3
"""
Setup script cho MRL Eye Dataset
"""
import os
import shutil


def create_directories():
    """Tạo cấu trúc thư mục cho MRL Eye Dataset"""
    dirs = [
        'models',
        'data/eyes/open',
        'data/eyes/closed'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("✅ Đã tạo cấu trúc thư mục")


def download_dataset():
    """Tải MRL Eye Dataset bằng kagglehub"""
    try:
        import kagglehub
        print("📥 Đang tải MRL Eye Dataset...")

        # Tải dataset
        path = kagglehub.dataset_download("imadeddinedjerarda/mrl-eye-dataset")
        print(f"✅ Đã tải dataset tại: {path}")
        print(f"Nội dung thư mục: {os.listdir(path)}")

        # Copy vào data/eyes
        if os.path.exists(path):
            for item in os.listdir(path):
                src_path = os.path.join(path, item)
                print(f"Kiểm tra: {item} - {os.path.isdir(src_path)}")

                if os.path.isdir(src_path):
                    # Tìm thư mục Open-Eyes và Close-Eyes trong mrleyedataset
                    if 'mrleyedataset' in item.lower():
                        for sub_item in os.listdir(src_path):
                            sub_path = os.path.join(src_path, sub_item)
                            if os.path.isdir(sub_path):
                                files = os.listdir(sub_path)
                                print(f"Thư mục {sub_item} có {len(files)} files")
                                
                                if 'Open-Eyes' in sub_item:
                                    dst_dir = 'data/eyes/open'
                                elif 'Close-Eyes' in sub_item:
                                    dst_dir = 'data/eyes/closed'
                                else:
                                    print(f"Bỏ qua: {sub_item}")
                                    continue
                                    
                                for file in files:
                                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                        src_file = os.path.join(sub_path, file)
                                        dst_file = os.path.join(dst_dir, file)
                                        try:
                                            shutil.copy2(src_file, dst_file)
                                        except Exception as e:
                                            print(f"Lỗi copy {file}: {e}")
                    else:
                        files = os.listdir(src_path)
                        print(f"Thư mục {item} có {len(files)} files")
                        
                        if 'open' in item.lower():
                            dst_dir = 'data/eyes/open'
                        elif 'close' in item.lower():
                            dst_dir = 'data/eyes/closed'
                        else:
                            print(f"Bỏ qua thư mục: {item}")
                            continue
                            
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                src_file = os.path.join(src_path, file)
                                dst_file = os.path.join(dst_dir, file)
                                try:
                                    shutil.copy2(src_file, dst_file)
                                except Exception as e:
                                    print(f"Lỗi copy {file}: {e}")

            # Kiểm tra kết quả
            open_count = len([f for f in os.listdir('data/eyes/open') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            closed_count = len([f for f in os.listdir('data/eyes/closed') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ Đã copy dataset: {open_count} ảnh mở, {closed_count} ảnh nhắm")
            return True

    except ImportError:
        print("❌ kagglehub chưa được cài đặt!")
        print("Chạy: pip install kagglehub")
        return False
    except Exception as e:
        print(f"❌ Lỗi tải dataset: {e}")
        print("📋 Cách khắc phục:")
        print("   1. Đăng nhập Kaggle: kagglehub.login()")
        print("   2. Hoặc tải thủ công từ Kaggle")
        return False


def main():
    """Main setup workflow"""
    print("🎯 Eye State Detection - Setup")
    print("=" * 50)

    # Create directories
    create_directories()

    # Download dataset
    print("\n📦 Dataset Setup")
    download_dataset()

    print("\n✅ Setup completed!")
    print("📋 Các bước tiếp theo:")
    print("   1. Chạy: python train.py (để train mô hình)")
    print("   2. Chạy: python app.py (để test giao diện)")



if __name__ == "__main__":
    main()