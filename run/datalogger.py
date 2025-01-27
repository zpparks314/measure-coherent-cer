import os

class DataLogger:
    def __init__(self, directory=None, filename="experiments_info.txt", id=None):
        """
        Initialize the DataLogger class.

        Args:
            directory (str, optional): The directory where the file will be stored. Defaults to None.
            filename (str, optional): The name of the file. Defaults to None.
            id (str, optional): The ID associated with the data logger. Defaults to None.
        """
        self.directory = directory
        self.filename = filename
        self.filepath = os.path.join(directory, filename) if directory and filename else None
        self.id = id

        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Create the file if it doesn't exist
        if self.filepath and not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as file:
                pass

    def write_data(self, data):
        """
        Write data to the file, only if the key-value pair is not already present.

        Args:
            data (dict): The data to be written to the file.
        """
        if self.filepath is None:
            raise ValueError("Filepath is not set. Please provide a valid directory and filename.")
        
        existing_data = self.read_data()
        
        with open(self.filepath, 'a') as file:
            for key, value in data.items():
                if key not in existing_data:
                    try:
                        file.write(f"{key}: {str(value)}\n")
                    except Exception as e:
                        raise ValueError(f"Failed to write key-value pair: {key}: {value}. Error: {str(e)}")

    def read_data(self):
        """
        Read the contents of the file.

        Returns:
            dict: A dictionary of key-value pairs.
        """
        if self.filepath is None:
            raise ValueError("Filepath is not set. Please provide a valid directory and filename.")
        with open(self.filepath, 'r') as file:
            data = {}
            for line in file:
                key, value = line.strip().split(': ')
                data[key] = value
            return data

    def clear_data(self, confirmation=False):
        """
        Clear the contents of the file if confirmation is provided.
        """
        if self.filepath is None:
            raise ValueError("Filepath is not set. Please provide a valid directory and filename.")
        
        existing_data = self.read_data()
        if existing_data:
            if confirmation:
                with open(self.filepath, 'w') as file:
                    pass
                print("Data cleared.")
            else:
                print(F"Data exists in the file {self.filepath}. First five key-value pairs:")
                print("\n".join([f"{key}: {value}" for key, value in list(existing_data.items())[:5]]))
                user_confirmation = input("Data exists in the file. Are you sure you want to clear it? (y/n): ")
                if user_confirmation.lower() == "y":
                    with open(self.filepath, 'w') as file:
                        pass
                    print("Data cleared.")
                else:
                    print("Clearing data operation cancelled.")
        else:
            print("No data exists in the file.")

    def check_inputs(self):
        """
        Check if filename, directory, or id are None.

        Returns:
            str: A message indicating which inputs are None.
        """
        inputs = []
        if self.filename is None:
            inputs.append("filename")
        if self.directory is None:
            inputs.append("directory")
        if self.id is None:
            inputs.append("id")
        if inputs:
            return f"The following inputs are None: {', '.join(inputs)}"
        else:
            return "All inputs are provided."
