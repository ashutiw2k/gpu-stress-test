import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch")

import torch
import time
import json
import argparse
import threading
import os
from datetime import datetime
from torch import nn

import pynvml  # nvidia-ml-py package — installs as the pynvml module
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================================================
# INIT GPU METRIC MONITOR
# ==============================================================

pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

gpu_stats = {
    "util": [],
    "mem": [],
    "temp": [],
    "power": [],
    "time": []
}

monitor_running = True


def gpu_monitor_thread(interval=0.25):
    start = time.time()
    while monitor_running:
        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        temp = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000  # watts

        gpu_stats["util"].append(util)
        gpu_stats["mem"].append(mem.used / mem.total * 100)
        gpu_stats["temp"].append(temp)
        gpu_stats["power"].append(power)
        gpu_stats["time"].append(time.time() - start)

        time.sleep(interval)


# ==============================================================
# REAL-TIME GPU GRAPH WINDOW
# ==============================================================

def start_realtime_graph():
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    fig.suptitle("GPU Real-Time Stress Test Metrics")

    def update(_):
        if len(gpu_stats["time"]) < 2:
            return

        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()

        axs[0].plot(gpu_stats["time"], gpu_stats["util"], label="GPU Util %", color="cyan")
        axs[0].set_ylabel("GPU Util %")
        axs[0].set_ylim(0, 100)

        axs[1].plot(gpu_stats["time"], gpu_stats["mem"], label="VRAM %", color="orange")
        axs[1].set_ylabel("VRAM %")
        axs[1].set_ylim(0, 100)

        axs[2].plot(gpu_stats["time"], gpu_stats["temp"], label="Temp (°C)", color="red")
        axs[2].set_ylabel("Temp °C")

        axs[3].plot(gpu_stats["time"], gpu_stats["power"], label="Power (W)", color="green")
        axs[3].set_ylabel("Power (W)")

        for ax in axs:
            ax.grid(True)

    ani = animation.FuncAnimation(fig, update, interval=300)
    plt.tight_layout()
    plt.show()


# ==============================================================
# STRESS TRAINING LOOP
# ==============================================================

def stress_test(epochs, batch_size, vram_fraction, log_file, result_dict):
    from torchvision.models import resnet50

    print("\nStarting ResNet50 Stress Test...")
    model = resnet50(weights=None).cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Run one full calibration step to measure real per-step peak overhead.
    # This accounts for Adam state init, forward activations, and gradients.
    torch.cuda.reset_peak_memory_stats(0)
    _x = torch.randn(batch_size, 3, 224, 224, device="cuda")
    _y = torch.randint(0, 1000, (batch_size,), device="cuda")
    loss_fn(model(_x), _y).backward()
    opt.step()
    opt.zero_grad()
    del _x, _y
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    current_allocated = torch.cuda.memory_allocated(0)
    peak_allocated = torch.cuda.max_memory_allocated(0)
    per_step_overhead = int((peak_allocated - current_allocated) * 1.5)  # 1.5x safety margin

    free_vram, _ = torch.cuda.mem_get_info(0)
    target_bytes = int(max(0, free_vram - per_step_overhead) * vram_fraction)
    bytes_per_img = 3 * 224 * 224 * 4

    dataset_size = max(batch_size, target_bytes // bytes_per_img)
    steps_per_epoch = dataset_size // batch_size

    print(f"Free VRAM after model+Adam init: {free_vram / 1024**3:.2f} GB")
    print(f"Per-step activation overhead:    {(peak_allocated - current_allocated) / 1024**3:.2f} GB (reserved {per_step_overhead / 1024**3:.2f} GB)")
    print(f"Dataset size: {dataset_size:,} images ({target_bytes / 1024**3:.2f} GB)")
    print(f"Steps per epoch: {steps_per_epoch}")

    x_data = torch.randn(dataset_size, 3, 224, 224, device="cuda")
    y_data = torch.randint(0, 1000, (dataset_size,), device="cuda")

    epoch_times = []
    all_losses = []

    for epoch in range(epochs):
        start = time.time()

        for i in range(steps_per_epoch):
            batch_x = x_data[i*batch_size:(i+1)*batch_size]
            batch_y = y_data[i*batch_size:(i+1)*batch_size]

            opt.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            opt.step()

            all_losses.append(float(loss))

        torch.cuda.synchronize()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        msg = f"Epoch {epoch+1}/{epochs} — {epoch_time:.2f} sec, {steps_per_epoch / epoch_time:.2f} steps/s"
        print(msg)
        log_file.write(msg + "\n")

    avg_epoch = sum(epoch_times) / len(epoch_times)

    result_dict.update({
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "avg_epoch_sec": avg_epoch,
        "steps_per_epoch": steps_per_epoch,
        "avg_steps_per_sec": steps_per_epoch / avg_epoch,
        "loss_curve": all_losses,
        "gpu_stats": gpu_stats
    })


# ==============================================================
# CLI ARGUMENTS
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RTX+class Stress Test Benchmark")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--vram", type=float, default=0.75,
                        help="Fraction of VRAM to fill with synthetic data")
    parser.add_argument("--nogui", action="store_true",
                        help="Disable real-time GPU graph")

    return parser.parse_args()


# ==============================================================
# MAIN
# ==============================================================

def main():
    args = parse_args()

    # Create logs
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/training_log_{timestamp}.txt"
    result_path = f"results/metrics_{timestamp}.json"

    log_file = open(log_path, "w")

    # Start GPU monitor thread
    global monitor_running
    monitor_running = True
    t = threading.Thread(target=gpu_monitor_thread)
    t.start()

    result_data = {}

    try:
        stress_test(
            epochs=args.epochs,
            batch_size=args.batch,
            vram_fraction=args.vram,
            log_file=log_file,
            result_dict=result_data
        )
    finally:
        monitor_running = False
        t.join()
        log_file.close()
        pynvml.nvmlShutdown()

    # Save results JSON
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"\nResults saved to: {result_path}")
    print(f"Log saved to: {log_path}")

    # Show real-time graph
    if not args.nogui:
        start_realtime_graph()


if __name__ == "__main__":
    main()
