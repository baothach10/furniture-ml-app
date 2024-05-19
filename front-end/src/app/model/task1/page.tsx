"use client"
import { handleImageSubmit } from '@/app/actions/submit-images';
import Image from 'next/image';

import React, { useState } from 'react';

const FileUploadForm = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [recommendations, setRecommendations] = useState<File[]>()


  const handleFileChange = (event: { target: { files: React.SetStateAction<null>[]; }; }) => {
    setSelectedFile(event.target.files[0]);
    setImageUrl(URL.createObjectURL(event.target.files[0]))
  };


  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    if (selectedFile) {
      const response = await handleImageSubmit(selectedFile);
      setLoading(false)
      setRecommendations(response.recommendations)
      console.log(response.recommendations)
    }
  }

  return (
    <div>
      <h2>File Upload</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        {!loading && <button type="submit">Upload</button>}
      </form>
      {
        imageUrl && (
          <Image width={200} height={200} src={imageUrl} alt='input_image' />
        )
      }
      <div className='flex flex-row'>
        {recommendations && recommendations.map((rec, index) => (
          <Image width={200} height={200} src={'http://localhost:8001/' + rec} alt='recomendations' key={index} />
        ))}
        ))
      </div>
    </div >
  );
};

export default FileUploadForm;

