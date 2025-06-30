import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def visualize_csv(csv_path: str):
    # Skip empty files
    if os.path.getsize(csv_path) == 0:
        print(f"Skipping empty CSV file: {csv_path}")
        return
    # Read CSV data
    df = pd.read_csv(csv_path)
    # Use CSV filename as output directory
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join('logs', 'images', base)
    os.makedirs(out_dir, exist_ok=True)

    # Group by env_id and plot
    for env in sorted(df['env_id'].unique()):
        dfe = df[df['env_id'] == env]
        steps = dfe['step']
        fig, axes = plt.subplots(4, 1, figsize=(8, 12))

        # Position x, y, z
        axes[0].plot(steps, dfe[['x', 'y', 'z']])
        axes[0].set_title(f'Env {env} - Position')
        axes[0].legend(['x','y','z'])

        # Velocity vx, vy, vz
        axes[1].plot(steps, dfe[['vx', 'vy', 'vz']])
        axes[1].set_title('Velocity')
        axes[1].legend(['vx','vy','vz'])

        # Thrust
        axes[2].plot(steps, dfe['thrust'], color='tab:orange')
        axes[2].set_title('Thrust')

        # Moment (moment_x, moment_y, moment_z)
        axes[3].plot(steps, dfe[['moment_x','moment_y','moment_z']])
        axes[3].set_title('Moment')
        axes[3].legend(['mx','my','mz'])

        for ax in axes:
            ax.set_xlabel('step')
            ax.grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{env}.png'))
        plt.close(fig)


def main():
    # Find all CSV files
    patterns = [os.path.join('logs','*.csv'), os.path.join('logs','**','*.csv')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    for f in files:
        print(f'Processing {f}')
        visualize_csv(f)


if __name__ == '__main__':
    main()
