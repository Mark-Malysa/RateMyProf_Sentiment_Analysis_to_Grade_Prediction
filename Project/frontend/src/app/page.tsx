import { PredictionForm } from "@/components/PredictionForm";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50 dark:bg-slate-950 font-sans selection:bg-indigo-500/30">

      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-12 md:py-24">

        {/* Header */}
        <div className="text-center space-y-4 mb-16">
          <div className="inline-block px-4 py-1.5 rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 font-medium text-sm mb-4">
            ✨ AI-Powered Grade Prediction
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 dark:text-white">
            Will You <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">Pass?</span>
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Combine professor reviews with your study habits to see your future grade.
            <br className="hidden md:block" /> Powered by BERT embeddings & Machine Learning.
          </p>
        </div>

        {/* Main Application */}
        <PredictionForm />

        {/* Footer */}
        <div className="mt-24 text-center text-slate-400 text-sm">
          <p>Built for the RateMyProf Sentiment Project • 2025</p>
        </div>
      </div>
    </main>
  );
}
