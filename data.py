import csv
import math
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_FILE = 'synchronized_lidar_odom.csv'
data = defaultdict(lambda: {'lidar': [], 'odom': None})

# Lecture du fichier
with open(CSV_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        t = float(row['timestamp'])
        lidar_x = float(row['lidar_x'])
        lidar_y = float(row['lidar_y'])
        odom_x = float(row['odom_x'])
        odom_y = float(row['odom_y'])
        theta = float(row['odom_theta'])

        # Sauvegarde odom seulement une fois par timestamp
        if data[t]['odom'] is None:
            data[t]['odom'] = (odom_x, odom_y, theta)

        data[t]['lidar'].append((lidar_x, lidar_y))

# Pour afficher la trajectoire du robot
trajectory_x = []
trajectory_y = []

plt.figure(figsize=(8, 8))

# Affichage boucle temporelle
for t, content in sorted(data.items()):
    odom = content['odom']
    lidar_pts = content['lidar']

    if odom is None:
        continue

    odom_x, odom_y, theta = odom
    trajectory_x.append(odom_x)
    trajectory_y.append(odom_y)

    # Transformation des points LiDAR en coordonnées globales
    transformed_lidar = []
    for lx, ly in lidar_pts:
        x_abs = odom_x + lx * math.cos(theta) - ly * math.sin(theta)
        y_abs = odom_y + lx * math.sin(theta) + ly * math.cos(theta)
        transformed_lidar.append((x_abs, y_abs))

    # Tracé
    plt.clf()
    plt.title(f"Instant t = {t:.2f} s")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')

    # Trajectoire du robot
    plt.plot(trajectory_x, trajectory_y, 'r--', label='Trajectoire')

    # Position actuelle du robot
    plt.plot(odom_x, odom_y, 'ro', label='Position robot')
    plt.arrow(
        odom_x, odom_y,
        0.3 * math.cos(theta), 0.3 * math.sin(theta),
        head_width=0.05, color='red'
    )

    # Points LiDAR
    if transformed_lidar:
        xs, ys = zip(*transformed_lidar)
        plt.scatter(xs, ys, s=5, c='blue', label='LiDAR')

    plt.legend()
    plt.pause(0.1)

plt.show()
