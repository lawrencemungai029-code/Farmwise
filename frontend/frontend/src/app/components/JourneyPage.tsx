import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Sprout, Droplets, TrendingUp, Wheat, Coins } from "lucide-react";

const stages = [
  {
    icon: <Sprout className="w-8 h-8 text-green-600" />, label: "Planting", message: "Start strong! Choose the best seeds for your region.",
  },
  {
    icon: <Droplets className="w-8 h-8 text-blue-500" />, label: "Irrigation", message: "Keep your crops hydrated. Monitor rainfall and irrigate as needed.",
  },
  {
    icon: <TrendingUp className="w-8 h-8 text-amber-500" />, label: "Growth", message: "Your crops are growing! Watch for pests and apply fertilizer wisely.",
  },
  {
    icon: <Wheat className="w-8 h-8 text-yellow-600" />, label: "Harvest", message: "Time to harvest! Plan for storage and market access.",
  },
  {
    icon: <Coins className="w-8 h-8 text-green-700" />, label: "Repayment", message: "Sell your produce and repay your loan. Congratulations!",
  },
];

export default function JourneyPage({ onFinish }: { onFinish: () => void }) {
  const [stage, setStage] = useState(0);
  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-green-50 to-yellow-50">
      <Card className="w-full max-w-lg shadow-lg">
        <CardHeader>
          <CardTitle>"Together Journey"</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center mb-4">
            {stages[stage].icon}
            <div className="text-xl font-semibold mt-2">{stages[stage].label}</div>
            <div className="text-gray-600 mb-2">{stages[stage].message}</div>
            <Progress value={((stage + 1) / stages.length) * 100} className="mb-2 w-full" />
            <div className="text-sm text-green-700">{stage < stages.length - 1 ? "You’re doing great!" : "Journey complete!"}</div>
          </div>
          <button
            className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700"
            onClick={() => stage < stages.length - 1 ? setStage(stage + 1) : onFinish()}
          >
            {stage < stages.length - 1 ? "Continue Journey" : "Finish"}
          </button>
        </CardContent>
      </Card>
    </div>
  );
}
