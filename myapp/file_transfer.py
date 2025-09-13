import os
import shutil

class FileTransfer:
    @staticmethod
    def move_output_to_static():
        """
        Move all files from the output directory to static/myapp/analysis/resaults.
        Creates the destination directory if it does not exist.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'output')
        dest_dir = os.path.join(base_dir, 'myapp', 'static', 'myapp', 'analysis', 'resaults')

        # f = open(os.path.join(output_dir, 'testo.txt'), "x")
        
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                src_path = os.path.join(output_dir, filename)
                dest_path = os.path.join(dest_dir, filename)
                if os.path.isfile(src_path):
                    shutil.move(src_path, dest_path)

if __name__ == "__main__":
    FileTransfer.move_output_to_static()
