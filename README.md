## Pick-place-model

## Pick-place-model

### Run code

To run the basic setup: franka arm robot with desk, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/janhq/pick-place-model
    ```
2. Navigate to the project directory:
    ```sh
    cd pick-place-model
    ```
3. Run the main script:
    ```sh
    python script/pick_place/pick_place.py
    ```

### adding different objects to the scene
1. Clone the repo that contain many xml 3d object
    ```sh
    https://github.com/vikashplus/furniture_sim.git
    ```
2. change the file path in scene.add_entity() of both microwave_door, slide_cabinet

3. Run the script:
    ```sh
    python script/pick_place/spawn_objects.py
    ```