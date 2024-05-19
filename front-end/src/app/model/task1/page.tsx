"use client"
import {handleTask1ImageSubmit} from '@/app/actions/submit-images';
import Image from 'next/image';

import React, {useState, useTransition} from 'react';

const FileUploadForm = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [imageUrl, setImageUrl] = useState(null)
    const [isPending, startTransition] = useTransition();
    const [predictedLabel, setPredictedLabel] = useState<string>('')

    const handleFileChange = (event: { target: { files: React.SetStateAction<null>[]; }; }) => {
        setSelectedFile(event.target.files[0]);
        setImageUrl(URL.createObjectURL(event.target.files[0]))
    };


    const onSubmit = (e) => {
      e.preventDefault()
        startTransition(async () => {
            if (selectedFile) {
                const response = await handleTask1ImageSubmit(selectedFile);
                setPredictedLabel(JSON.stringify(response))
            }
        })
    }
    return (
        <div>
            <h2>File Upload</h2>
            <form onSubmit={onSubmit}>
                <input type="file" onChange={handleFileChange}/>
                {!isPending && <button type="submit">Upload</button>}
            </form>
            {
                imageUrl && (
                    <Image width={200} height={200} src={imageUrl} alt='input_image'/>
                )
            }
            <div className='flex flex-row'>
                <p>Model predict the image as: {predictedLabel}</p>
            </div>
        </div>
    );
};

export default FileUploadForm;

