import os
import atexit
import shutil

CACHED_PATH = "~\.cache\huggingface\hub\models--facebook--blenderbot-400M-distill"

class Delete_Cached_Model():
    def delete_model_files(self):
        cache_dir = os.path.expanduser(CACHED_PATH)
        if os.path.exists(cache_dir):
            user_in = input("Do you want to delete the model files stored in cache dir? (y/n): ")
            if user_in.lower() == 'y':
                shutil.rmtree(cache_dir)
                print("Deleted model files from cached directory!")
            else:
                print("Model files not deleted.")
        else:
            print(f"Model files not found in path: {cache_dir}")

    def register_on_quit(self):
        atexit.register(self.delete_model_files)

if __name__ == "__main__":
    pass
