import React from 'react'
import { motion } from 'framer-motion'
import { Shield, Moon, Sun } from 'lucide-react'

const Header = ({ darkMode, toggleDarkMode }) => {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-effect border-b border-white/20 dark:border-gray-700/50"
    >
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold gradient-text">AI Face Detector</span>
        </div>
        
        <button
          onClick={toggleDarkMode}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          {darkMode ? (
            <Sun className="w-5 h-5 text-yellow-500" />
          ) : (
            <Moon className="w-5 h-5 text-gray-600" />
          )}
        </button>
      </div>
    </motion.header>
  )
}

export default Header