import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { Upload, Image, X, Zap } from 'lucide-react'

const ImageUploader = ({ onAnalysis, loading }) => {
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0]
    if (selectedFile) {
      setFile(selectedFile)
      const reader = new FileReader()
      reader.onload = () => setPreview(reader.result)
      reader.readAsDataURL(selectedFile)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false
  })

  const handleAnalyze = () => {
    if (file) {
      onAnalysis(file)
    }
  }

  const clearImage = () => {
    setPreview(null)
    setFile(null)
  }

  return (
    <div className="space-y-6">
      {!preview ? (
        <motion.div
          {...getRootProps()}
          className={`upload-zone cursor-pointer ${
            isDragActive ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-900/20' : ''
          }`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <input {...getInputProps()} />
          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <div>
              <p className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                {isDragActive ? 'Drop your image here' : 'Upload an image to analyze'}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Supports JPG, PNG, WebP • Max 10MB
              </p>
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative"
        >
          <div className="relative rounded-2xl overflow-hidden bg-gray-100 dark:bg-gray-800">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-64 object-cover"
            />
            <button
              onClick={clearImage}
              className="absolute top-4 right-4 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          
          <div className="mt-6 flex justify-center">
            <motion.button
              onClick={handleAnalyze}
              disabled={loading}
              className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold flex items-center space-x-2 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Zap className="w-5 h-5" />
              <span>{loading ? 'Analyzing...' : 'Analyze Image'}</span>
            </motion.button>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default ImageUploader