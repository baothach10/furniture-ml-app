export  const handleImageSubmit = async (image: File) => {
  const formData = new FormData();
  formData.append("file", image);
  const response = await fetch("http://localhost:8001/task2", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    return { error: { message: `HTTP error! status: ${response.status}` } };
  }

  return await response.json();
};