class FileManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        try:
            with open(self.file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return "File not found."
        except Exception as e:
            return f"An error occurred: {e}"

    def write_file(self, content):
        try:
            with open(self.file_path, 'w') as file:
                file.write(content)
            return "File written successfully."
        except Exception as e:
            return f"An error occurred: {e}"