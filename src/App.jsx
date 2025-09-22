import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Zap, Shield, Eye, Moon, Sun } from 'lucide-react'
import ImageUploader from './components/ImageUploader'
import ResultDisplay from './components/ResultDisplay'
import Header from './components/Header'
import Features from './components/Features'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(false)

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
    document.documentElement.classList.toggle('dark')
  }

  const handleAnalysis = useCallback(async (file) => {
    setLoading(true)
    setResult(null)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) throw new Error('Analysis failed')
      
      const data = await response.json()
      setResult(data)
    } catch (error) {
      setResult({
        error: true,
        message: 'Failed to analyze image. Please try again.'
      })
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className="min-h-screen">
      <Header darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
      
      <main className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold mb-4">
            <span className="gradient-text">AI Face Detector</span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Advanced AI-powered detection of artificially generated faces with 95%+ accuracy
          </p>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="glass-effect rounded-3xl p-8 mb-8"
          >
            <ImageUploader 
              onAnalysis={handleAnalysis}
              loading={loading}
            />
          </motion.div>

          <AnimatePresence>
            {(result || loading) && (
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -50 }}
                transition={{ duration: 0.5 }}
              >
                <ResultDisplay result={result} loading={loading} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <Features />
      </main>
    </div>
  )
}

export default App