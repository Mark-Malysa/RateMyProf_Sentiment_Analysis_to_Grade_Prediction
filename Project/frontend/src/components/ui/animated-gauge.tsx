"use client"

import { motion } from "framer-motion"

interface AnimatedGaugeProps {
    value: number // 0 to 4.0
    grade: string
}

export function AnimatedGauge({ value, grade }: AnimatedGaugeProps) {
    // Normalize 0-4.0 to 0-100 for circle progress
    const percentage = (value / 4.0) * 100
    const circumference = 2 * Math.PI * 45 // radius 45

    // Color selection based on grade
    const getColor = (g: string) => {
        if (g.startsWith('A')) return '#22c55e' // Green-500
        if (g.startsWith('B')) return '#3b82f6' // Blue-500
        if (g.startsWith('C')) return '#eab308' // Yellow-500
        if (g.startsWith('D')) return '#f97316' // Orange-500
        return '#ef4444' // Red-500
    }

    const color = getColor(grade)

    return (
        <div className="relative flex items-center justify-center w-48 h-48 mx-auto">
            {/* Background Circle */}
            <svg className="absolute w-full h-full transform -rotate-90">
                <circle
                    cx="96"
                    cy="96"
                    r="45"
                    className="text-slate-200 dark:text-slate-800"
                    strokeWidth="12"
                    stroke="currentColor"
                    fill="transparent"
                />
                {/* Animated Progress Circle */}
                <motion.circle
                    cx="96"
                    cy="96"
                    r="45"
                    stroke={color}
                    strokeWidth="12"
                    fill="transparent"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: circumference - (percentage / 100) * circumference }}
                    transition={{ duration: 1.5, ease: "easeOut", delay: 0.2 }}
                />
            </svg>

            {/* Center Text */}
            <div className="flex flex-col items-center">
                <motion.span
                    className="text-5xl font-bold text-slate-900 dark:text-white"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.5 }}
                >
                    {grade}
                </motion.span>
                <motion.span
                    className="text-sm font-medium text-slate-500 dark:text-slate-400"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.8 }}
                >
                    GPA: {value.toFixed(2)}
                </motion.span>
            </div>
        </div>
    )
}
