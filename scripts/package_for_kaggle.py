import os
import zipfile
import glob

def package_project():
    output_filename = "kaggle_codebase.zip"
    
    # Directories and files to include
    include_paths = [
        "configs",
        "scripts",
        "src",
        "requirements.txt"
    ]
    
    # Exclude patterns
    exclude_dirs = ["__pycache__", ".ipynb_checkpoints"]
    exclude_extensions = [".pyc", ".pyo", ".pyd"]
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for base_path in include_paths:
            if not os.path.exists(base_path):
                print(f"Warning: {base_path} not found, skipping.")
                continue
                
            if os.path.isfile(base_path):
                zipf.write(base_path, base_path)
                print(f"Added file: {base_path}")
            else:
                for root, dirs, files in os.walk(base_path):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        if any(file.endswith(ext) for ext in exclude_extensions):
                            continue
                            
                        file_path = os.path.join(root, file)
                        # Archive path should be relative to the current directory
                        arcname = file_path
                        zipf.write(file_path, arcname)
                        print(f"Added: {arcname}")
                        
    print(f"\nSuccessfully packaged codebase to: {os.path.abspath(output_filename)}")
    print(f"Upload {output_filename} as a dataset to Kaggle.")

if __name__ == "__main__":
    # Ensure script runs from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    package_project()
