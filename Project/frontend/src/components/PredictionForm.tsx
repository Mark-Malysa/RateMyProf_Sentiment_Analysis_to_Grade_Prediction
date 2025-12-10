"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Slider } from "@/components/ui/slider"
import { Loader2, GraduationCap, BookOpen, Brain, Star, Gauge } from "lucide-react"
import { AnimatedGauge } from "@/components/ui/animated-gauge"

interface PredictionResult {
    combined_gpa: number
    combined_grade: string
    recommendation: string
    breakdown: {
        review_based: { gpa: number, grade: string, weight: number }
        habits_based: { gpa: number, grade: string, weight: number }
    }
}

export function PredictionForm() {
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)

    // Form State
    const [rating, setRating] = useState([3.0])
    const [difficulty, setDifficulty] = useState([3.0])
    const [review, setReview] = useState("")
    const [studyHours, setStudyHours] = useState([4.0])
    const [priorGpa, setPriorGpa] = useState([3.0])
    const [motivation, setMotivation] = useState([5])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setResult(null)

        try {
            const response = await fetch("http://localhost:8000/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    rating: rating[0],
                    difficulty: difficulty[0],
                    review_text: review,
                    study_hours: studyHours[0],
                    prior_gpa: priorGpa[0],
                    motivation: motivation[0]
                }),
            })

            const data = await response.json()
            setResult(data)
        } catch (error) {
            console.error("Prediction failed:", error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="w-full max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">

                {/* INPUT COLUMN */}
                <motion.div
                    className="space-y-6 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl p-8 rounded-3xl shadow-xl border border-white/20"
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="space-y-2">
                        <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400">
                            Class Details
                        </h2>
                        <p className="text-slate-500 dark:text-slate-400 text-sm">Tell us about the professor and course.</p>
                    </div>

                    <div className="space-y-6">
                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <Star className="w-4 h-4 text-yellow-500" /> Professor Rating
                                </label>
                                <span className="font-bold text-indigo-600 dark:text-indigo-400">{rating[0]} / 5</span>
                            </div>
                            <Slider value={rating} onValueChange={setRating} min={1} max={5} step={0.1} />
                        </div>

                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <Gauge className="w-4 h-4 text-red-500" /> Difficulty
                                </label>
                                <span className="font-bold text-indigo-600 dark:text-indigo-400">{difficulty[0]} / 5</span>
                            </div>
                            <Slider value={difficulty} onValueChange={setDifficulty} min={1} max={5} step={0.1} />
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Review Text (Optional)</label>
                            <textarea
                                className="w-full p-3 rounded-xl bg-slate-50 dark:bg-slate-800 border-none focus:ring-2 focus:ring-indigo-500 min-h-[80px]"
                                placeholder="Paste a review or describe the professor..."
                                value={review}
                                onChange={(e) => setReview(e.target.value)}
                            />
                        </div>

                        <div className="h-px bg-slate-200 dark:bg-slate-800" />

                        <div className="space-y-2">
                            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-600 to-emerald-600 dark:from-teal-400 dark:to-emerald-400">
                                Your Habits
                            </h2>
                            <p className="text-slate-500 dark:text-slate-400 text-sm">Be honest for the best prediction!</p>
                        </div>

                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <BookOpen className="w-4 h-4 text-teal-500" /> Daily Study Hours
                                </label>
                                <span className="font-bold text-teal-600 dark:text-teal-400">{studyHours[0]} hrs</span>
                            </div>
                            <Slider value={studyHours} onValueChange={setStudyHours} min={0} max={12} step={0.5} />
                        </div>

                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <GraduationCap className="w-4 h-4 text-purple-500" /> Prior GPA
                                </label>
                                <span className="font-bold text-purple-600 dark:text-purple-400">{priorGpa[0].toFixed(2)}</span>
                            </div>
                            <Slider value={priorGpa} onValueChange={setPriorGpa} min={0} max={4} step={0.01} />
                        </div>

                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <Brain className="w-4 h-4 text-pink-500" /> Motivation Level
                                </label>
                                <span className="font-bold text-pink-600 dark:text-pink-400">{motivation[0]} / 10</span>
                            </div>
                            <Slider value={motivation} onValueChange={setMotivation} min={1} max={10} step={1} />
                        </div>
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={loading}
                        className="w-full py-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold text-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                        {loading ? <Loader2 className="animate-spin" /> : "Predict Grade"}
                    </button>
                </motion.div>

                {/* RESULTS COLUMN */}
                <div className="flex flex-col gap-6">
                    <AnimatePresence mode="wait">
                        {!result ? (
                            <motion.div
                                className="h-full flex items-center justify-center bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm rounded-3xl border border-dashed border-slate-300 dark:border-slate-700"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                            >
                                <div className="text-center text-slate-400 p-8">
                                    <SparklesIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
                                    <p>Enter details to see your predicted grade</p>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                className="space-y-6"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5 }}
                            >
                                {/* Main Result Card */}
                                <div className="bg-white dark:bg-slate-900 rounded-3xl p-8 shadow-2xl border border-indigo-100 dark:border-indigo-900/30 overflow-hidden relative">
                                    <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500" />

                                    <h3 className="text-center font-medium text-slate-500 uppercase tracking-widest text-xs mb-8">
                                        Predicted Performance
                                    </h3>

                                    <AnimatedGauge value={result.combined_gpa} grade={result.combined_grade} />

                                    <motion.div
                                        className="mt-8 p-4 rounded-xl bg-slate-50 dark:bg-slate-800 text-center"
                                        initial={{ scale: 0.9, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        transition={{ delay: 0.6 }}
                                    >
                                        <p className="font-medium text-slate-800 dark:text-slate-200">
                                            {result.recommendation}
                                        </p>
                                    </motion.div>
                                </div>

                                {/* Breakdown Cards */}
                                <div className="grid grid-cols-2 gap-4">
                                    <motion.div
                                        className="p-4 rounded-2xl bg-white/80 dark:bg-slate-900/80 shadow-lg border border-purple-100 dark:border-purple-900/20"
                                        initial={{ x: 20, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        transition={{ delay: 0.8 }}
                                    >
                                        <p className="text-xs text-slate-500 uppercase font-bold mb-1">Student Habits</p>
                                        <p className="text-2xl font-bold text-slate-800 dark:text-white">
                                            {result.breakdown.habits_based.grade}
                                            <span className="text-sm font-normal text-slate-400 ml-1">
                                                ({result.breakdown.habits_based.gpa.toFixed(2)})
                                            </span>
                                        </p>
                                    </motion.div>

                                    <motion.div
                                        className="p-4 rounded-2xl bg-white/80 dark:bg-slate-900/80 shadow-lg border border-indigo-100 dark:border-indigo-900/20"
                                        initial={{ x: 20, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        transition={{ delay: 0.9 }}
                                    >
                                        <p className="text-xs text-slate-500 uppercase font-bold mb-1">Prof Rating</p>
                                        <p className="text-2xl font-bold text-slate-800 dark:text-white">
                                            {result.breakdown.review_based.grade}
                                            <span className="text-sm font-normal text-slate-400 ml-1">
                                                ({result.breakdown.review_based.gpa.toFixed(2)})
                                            </span>
                                        </p>
                                    </motion.div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    )
}

function SparklesIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            {...props}
        >
            <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
            <path d="M5 3v4" />
            <path d="M9 3v4" />
            <path d="M5 7h4" />
        </svg>
    )
}
