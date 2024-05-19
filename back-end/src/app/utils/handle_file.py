# Add binary files
def save_to_FS(type: str, file_name: str, extension: str, file_content: bytes):
    generated_name = f"./static/{type}/{file_name}"
    # generated_name = f"/var/static/{type}/{file_name}.{extension}"
    with open(generated_name, "wb") as file:
        file.write(file_content)
    file.close()
