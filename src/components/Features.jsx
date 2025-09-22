import React from 'react'
import { motion } from 'framer-motion'
import { Zap, Shield, Eye, Brain, Clock, Target } from 'lucide-react'

const features = [
  {
    icon: Target,
    title: '95%+ Accuracy',
    description: 'State-of-the-art deep learning models trained on millions of images'
  },
  {
    icon: Zap,
    title: 'Lightning Fast',
    description: 'Get results in under 2 seconds with optimized processing'
  },
  {
    icon: Shield,
    title: 'Privacy First',
    description: 'Images are processed locally and never stored on our servers'
  },
  {
    icon: Brain,
    title: 'Advanced AI',
    description: 'Detects patterns from Midjourney, DALL-E, Stable Diffusion, and more'
  },
  {
    icon: Eye,
    title: 'Detailed Analysis',
    description: 'Comprehensive breakdown of AI artifacts and visual inconsistencies'
  },
  {
    icon: Clock,
    title: 'Real-time Processing',
    description: 'Instant feedback with confidence scores and detailed metrics'
  }
]

const Features = () => {
  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="mt-16"
    >
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold gradient-text mb-4">
          Why Choose Our AI Detector?
        </h2>
        <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
          Built with cutting-edge technology and trained on the most comprehensive dataset available
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 * index }}
            className="glass-effect rounded-2xl p-6 hover:shadow-xl transition-all duration-300"
          >
            <div className="flex items-center space-x-4 mb-4">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                {feature.title}
              </h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400">
              {feature.description}
            </p>
          </motion.div>
        ))}
      </div>
    </motion.section>
  )
}

export default Features