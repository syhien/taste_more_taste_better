from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import sys
import requests


def inpaint_save(
    host: str,
    params: dict,
    input_image: bytes,
    input_mask: bytes = None,
    save_path: str = None,
) -> None:
    try:
        # response = requests.post(
        #     url=f"{host}/v1/generation/image-inpaint-outpaint",
        #     data=params,
        #     files={"input_image": input_image, "input_mask": input_mask},
        # ).json()
        # url = response[0]["url"]
        # out_img = requests.get(url).content
        # with open(save_path, "wb") as f:
        #     f.write(out_img)

        # ----- Your Implementation Here -----
        pass
        # ------------------------------------
    except Exception as e:
        print(e)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    dirs = os.listdir(input_dir)
    random.shuffle(dirs)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for sample_dir in dirs:
            sample_path = os.path.join(input_dir, sample_dir)
            if not os.path.isdir(sample_path):
                continue

            img_path = os.path.join(sample_path, f"{sample_dir}.jpg")
            mask_path = os.path.join(sample_path, f"{sample_dir}_mask.jpg")
            with open(os.path.join(sample_path, "inpaint.json"), "r") as f:
                config = json.load(f)

            host = config["host"]
            prompt = config["prompt"]
            neg_prompt = config["negative_prompt"]
            params = {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "async_process": False,
            }
            os.makedirs(os.path.join(output_dir, sample_dir), exist_ok=True)
            with open(img_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                executor.submit(
                    inpaint_save,
                    host,
                    params,
                    img_file.read(),
                    mask_file.read(),
                    os.path.join(output_dir, sample_dir, f"{sample_dir}.jpg"),
                )
