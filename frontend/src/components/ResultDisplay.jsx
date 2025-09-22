import React from 'react'
import { motion } from 'framer-motion'
import { CheckCircle, AlertTriangle, Loader, Eye, Brain, Clock } from 'lucide-react'

const ResultDisplay = ({ result, loading }) => {
  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="result-card text-center"
      >
        <div className="flex flex-col items-center space-y-4">
          <div className="relative">
            <Loader className="w-12 h-12 text-blue-500 animate-spin" />
            <div className="absolute inset-0 w-12 h-12 border-4 border-blue-200 rounded-full animate-pulse-slow"></div>
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200">
              Analyzing Image...
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Our AI is examining the image for artificial generation patterns
            </p>
          </div>
        </div>
      </motion.div>
    )
  }

  if (result?.error) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="result-card border-l-4 border-red-500"
      >
        <div className="flex items-center space-x-3">
          <AlertTriangle className="w-8 h-8 text-red-500" />
          <div>
            <h3 className="text-lg font-semibold text-red-600 dark:text-red-400">
              Analysis Failed
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              {result.message}
            </p>
          </div>
        </div>
      </motion.div>
    )
  }

  if (!result) return null

  const isAI = result.prediction === 'AI Generated'
  const confidence = Math.round(result.confidence * 100)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Main Result */}
      <div className={`result-card border-l-4 ${
        isAI ? 'border-red-500' : 'border-green-500'
      }`}>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            <div className={`p-3 rounded-full ${
              isAI ? 'bg-red-100 dark:bg-red-900/30' : 'bg-green-100 dark:bg-green-900/30'
            }`}>
              {isAI ? (
                <AlertTriangle className="w-8 h-8 text-red-500" />
              ) : (
                <CheckCircle className="w-8 h-8 text-green-500" />
              )}
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
                {result.prediction}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                {isAI ? 'This image appears to be artificially generated' : 'This image appears to be authentic'}
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-3xl font-bold text-blue-600">
              {confidence}%
            </div>
            <div className="text-sm text-gray-500">Confidence</div>
          </div>
        </div>
      </div>

      {/* Detailed Analysis */}
      <div className="grid md:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-4"
        >
          <div className="flex items-center space-x-3 mb-3">
            <Brain className="w-6 h-6 text-purple-500" />
            <h4 className="font-semibold text-gray-800 dark:text-gray-200">
              AI Patterns
            </h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Artifacts</span>
              <span className="text-sm font-medium">{result.artifacts || 'Low'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Consistency</span>
              <span className="text-sm font-medium">{result.consistency || 'High'}</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-4"
        >
          <div className="flex items-center space-x-3 mb-3">
            <Eye className="w-6 h-6 text-blue-500" />
            <h4 className="font-semibold text-gray-800 dark:text-gray-200">
              Visual Analysis
            </h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Texture</span>
              <span className="text-sm font-medium">{result.texture || 'Natural'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Symmetry</span>
              <span className="text-sm font-medium">{result.symmetry || 'Normal'}</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-4"
        >
          <div className="flex items-center space-x-3 mb-3">
            <Clock className="w-6 h-6 text-green-500" />
            <h4 className="font-semibold text-gray-800 dark:text-gray-200">
              Processing
            </h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Time</span>
              <span className="text-sm font-medium">{result.processing_time || '1.2s'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Model</span>
              <span className="text-sm font-medium">CNN-v2.1</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Confidence Bar */}
      <div className="glass-effect rounded-xl p-4">
        <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
          Confidence Distribution
        </h4>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600 dark:text-gray-400">AI Generated</span>
              <span className="font-medium">{isAI ? confidence : 100 - confidence}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full transition-all duration-1000"
                style={{ width: `${isAI ? confidence : 100 - confidence}%` }}
              ></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600 dark:text-gray-400">Real/Authentic</span>
              <span className="font-medium">{isAI ? 100 - confidence : confidence}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                style={{ width: `${isAI ? 100 - confidence : confidence}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default ResultDisplay