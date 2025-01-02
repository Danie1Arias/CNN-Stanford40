import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
import torch


class StanfordDataLoader:
    def __init__(self, base_dir="./Stanford40"):
        # Initialize the directories based on the base directory
        self.BASE_DIR = base_dir
        self.IMAGE_DIR = os.path.join(self.BASE_DIR, "JPEGImages")
        self.ANNOTATIONS_DIR = os.path.join(self.BASE_DIR, "XMLAnnotations")
        self.SPLITS_DIR = os.path.join(self.BASE_DIR, "ImageSplits")

    def read_actions(self, file_path):
        # Reads the actions and the number of images for each action from the given file.
        actions = {}
        with open(file_path, 'r') as file:
            for line in file.readlines()[1:]:  # Skip the header line.
                action, num_images = line.strip().split()
                actions[action] = int(num_images)  # Store action and its count in the dictionary.
        return actions

    def read_split(self, file_path):
        # Reads the list of image filenames from the split file (train/test).
        images = []
        with open(file_path, 'r') as file:
            for line in file:
                images.append(line.strip())  # Add each filename to the list.
        return images

    def parse_annotation(self, xml_file):
        # Parses an XML file to extract bounding box, action, and filename details.
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')  # Locate the bounding box information.
        action = root.find('.//action').text  # Extract the action label.
        filename = root.find('.//filename').text  # Extract the image filename.

        # Create a dictionary containing the bounding box details.
        box = {
            "xmin": int(bndbox.find('xmin').text),
            "ymin": int(bndbox.find('ymin').text),
            "xmax": int(bndbox.find('xmax').text),
            "ymax": int(bndbox.find('ymax').text),
        }
        return {"filename": filename, "action": action, "bndbox": box}

    def prepare_dataset(self):
        # Prepares the dataset by reading and organizing data from the splits and XML files.
        actions_file = os.path.join(self.SPLITS_DIR, "actions.txt")
        actions = self.read_actions(actions_file)  # Read the action names and their counts.

        # Initialize the dataset dictionary for train and test splits.
        dataset = {"train": [], "test": []}

        for action in actions.keys():
            # Get the paths to the train and test split files for the current action.
            train_file = os.path.join(self.SPLITS_DIR, f"{action}_train.txt")
            test_file = os.path.join(self.SPLITS_DIR, f"{action}_test.txt")

            # Read the filenames for the train and test splits.
            train_images = self.read_split(train_file)
            test_images = self.read_split(test_file)

            # Process each image in the train and test splits.
            for image_set, split in [(train_images, "train"), (test_images, "test")]:
                for image in image_set:
                    # Find the corresponding XML annotation file.
                    xml_file = os.path.join(self.ANNOTATIONS_DIR, image.replace('.jpg', '.xml'))
                    
                    if os.path.exists(xml_file):  # Only process if the XML file exists.
                        annotation = self.parse_annotation(xml_file)

                        # Append the processed image data to the appropriate split.
                        dataset[split].append({
                            "filename": os.path.join(self.IMAGE_DIR, annotation["filename"]),
                            "action": annotation["action"],  # Keep the action as a string
                            "bndbox": annotation["bndbox"]
                        })
        return dataset

    def create_dataloaders(self, transform, batch_size=32):
        # Crear un mapeo de acción a índice
        dataset = self.prepare_dataset()
        actions = sorted(set(item["action"] for split in dataset.values() for item in split))
        class_to_idx = {action: idx for idx, action in enumerate(actions)}

        # Custom Dataset with bounding box support
        class Stanford40DatasetWithBoxes(Dataset):
            def __init__(self, dataset, split, class_to_idx, transform=None):
                self.data = dataset[split]
                self.class_to_idx = class_to_idx
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                image = Image.open(item["filename"]).convert("RGB")
                
                # Convertir la acción a índice
                label = self.class_to_idx[item["action"]]
                
                # Recortar la imagen usando el bounding box
                bndbox = item["bndbox"]
                image = image.crop((bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]))
                
                # Aplicar transformaciones
                if self.transform:
                    image = self.transform(image)

                # Devolver la imagen y el label como tensor
                return image, torch.tensor(label, dtype=torch.long)

        # Crear datasets de entrenamiento y prueba
        train_dataset = Stanford40DatasetWithBoxes(dataset, split="train", class_to_idx=class_to_idx, transform=transform)
        test_dataset = Stanford40DatasetWithBoxes(dataset, split="test", class_to_idx=class_to_idx, transform=transform)

        # Crear DataLoaders usando Torch's DataLoader
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
