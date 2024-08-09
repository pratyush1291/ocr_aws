import torch

def check_cuda():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__ == "__main__":
    check_cuda()
