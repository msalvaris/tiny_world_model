from PIL import Image, ImageDraw
import numpy as np
import imageio
import os
import fire


def create_ball_image(position, image_size=(64, 64), ball_size=3):
    """Creates an image with a ball at the given position."""
    # Create a blank image
    image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(image)

    # Calculate ball bounds
    top_left = (position[0] - ball_size // 2, position[1] - ball_size // 2)
    bottom_right = (position[0] + ball_size // 2 + 1, position[1] + ball_size // 2 + 1)

    # Draw the ball
    draw.ellipse([top_left, bottom_right], fill=255)

    return image


def update_position(position, velocity, image_size):
    """Update ball position and velocity for bouncing."""
    new_position = [position[0] + velocity[0], position[1] + velocity[1]]

    # Check for collisions and bounce
    if new_position[0] <= 0 or new_position[0] >= image_size[0] - 1:
        velocity[0] = -velocity[0]
    if new_position[1] <= 0 or new_position[1] >= image_size[1] - 1:
        velocity[1] = -velocity[1]

    # Correct position if out of bounds
    new_position = [max(1, min(image_size[0] - 2, new_position[0])),
                    max(1, min(image_size[1] - 2, new_position[1]))]

    return new_position, velocity


def simulate_bouncing_ball(frames=10, image_size=(64, 64), position = (32,32), velocity = (1, -1)):

    position = list(position)
    velocity = list(velocity)
    
    for _ in range(frames):
        img = create_ball_image(position, image_size)
        yield img
        position, velocity = update_position(position, velocity, image_size)


def save_sequence(save_folder, frames=10, image_size=(64, 64), position = (32,32), velocity = (1, -1)):
    os.makedirs(save_folder, exist_ok=True)
    for i, img in enumerate(simulate_bouncing_ball(frames=frames, image_size=image_size, position = position, velocity = velocity )):
        img.save(os.path.join(save_folder,f"frame_{i}.png"))


def create_dataset(save_path, num_examples, random_seed=42):
    os.makedirs(save_path, exist_ok=True)
    num_frames = 20
    np.random.seed(42)
    position_x = np.random.randint(4,60, size=num_examples)
    position_y = np.random.randint(4,60, size=num_examples)
    velocity_x = np.random.randint(-4,4, size=num_examples)
    velocity_y = np.random.randint(-4,4, size=num_examples)

    for i in range(num_examples):
        save_folder = os.path.join(save_path, f"sequence_{i}")
        position = (position_x[i],position_y[i])
        velocity = (velocity_x[i],velocity_y[i])
        save_sequence(save_folder, frames=num_frames, image_size=(64, 64), position = position, velocity = velocity)


def create_movie():

    # Create a writer object
    writer = imageio.get_writer('bouncing_ball_movie.mp4', fps=10)  # Adjust the fps as needed

    # Simulate bouncing ball
    # Write each frame to the video
    for img in simulate_bouncing_ball(frames=400):
        writer.append_data(np.array(img))

    # Close the writer
    writer.close()

    print("Movie created successfully.")


def cli():
    fire.Fire({
        "ball": create_dataset
    })

    
if __name__=="__main__":
    # create_movie()
    create_dataset("/home/mat/ball_dataset", 1000)
