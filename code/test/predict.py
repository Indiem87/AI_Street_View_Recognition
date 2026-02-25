import os
import sys
import csv
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../train'))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from model import SVHN_Model1


class SVHNTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted([
            x for x in os.listdir(img_dir)
            if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name


def locate_test_dir(tcdata_dir):
    candidates = [
        os.path.join(tcdata_dir, 'mchar_test_a', 'mchar_test_a'),
        os.path.join(tcdata_dir, 'mchar_test_a')
    ]
    for path in candidates:
        if not os.path.isdir(path):
            continue
        for file_name in os.listdir(path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                return path
    raise FileNotFoundError('Cannot find test image directory under tcdata/mchar_test_a')


def digits_from_prediction(pred_row):
    digits = [str(x) for x in pred_row if x != 10]
    if not digits:
        return '0'
    return ''.join(digits)


def main():
    parser = argparse.ArgumentParser(description='SVHN street-view test prediction')

    workspace_root = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
    default_tcdata = os.path.join(workspace_root, 'tcdata')
    default_model = os.path.join(workspace_root, 'user_data', 'model_data', 'best_model.pth')
    default_output = os.path.join(workspace_root, 'prediction_result', 'result.tsv')

    parser.add_argument('--tcdata-dir', default=default_tcdata, help='Path to tcdata directory')
    parser.add_argument('--model-path', default=default_model, help='Path to model checkpoint')
    parser.add_argument('--output', default=default_output, help='Output TSV path')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    test_dir = locate_test_dir(args.tcdata_dir)

    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = SVHNTestDataset(test_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f'No test images found in: {test_dir}')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVHN_Model1().to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    results = []

    with torch.no_grad():
        for images, img_names in tqdm(loader, desc='Predict'):
            images = images.to(device)

            c1, c2, c3, c4, c5, c6 = model(images)
            preds = torch.stack([
                c1.argmax(dim=1),
                c2.argmax(dim=1),
                c3.argmax(dim=1),
                c4.argmax(dim=1),
                c5.argmax(dim=1),
                c6.argmax(dim=1),
            ], dim=1)

            preds = preds.cpu().tolist()
            for name, pred_row in zip(img_names, preds):
                code = digits_from_prediction(pred_row)
                results.append((name, code))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['file_name', 'file_code'])
        writer.writerows(results)

    print(f'Test directory: {test_dir}')
    print(f'Total images: {len(results)}')
    print(f'Result saved to: {args.output}')


if __name__ == '__main__':
    main()
