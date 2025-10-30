import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

function getTheme(score: number) {
  if (score >= 75) return { color: "green", label: "Eligible for support" };
  if (score >= 50) return { color: "amber", label: "Needs monitoring" };
  return { color: "red", label: "Requires further review" };
}

export default function Dashboard({ result, onContinue }: { result: any, onContinue: () => void }) {
  const { color, label } = getTheme((result.credit_score || 0) * 100);
  return (
    <div className="flex min-h-screen items-center justify-center bg-white">
      <Card className="w-full max-w-lg shadow-lg">
        <CardHeader>
          <CardTitle>Credit Score Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-4">
            <span className={`text-4xl font-bold text-${color}-600`}>{Math.round((result.credit_score || 0) * 100)}</span>
            <span className="text-lg">/ 100</span>
            <Badge variant={color}>{label}</Badge>
          </div>
          <Progress value={(result.credit_score || 0) * 100} className={`mb-4 bg-${color}-100`} />
          <div className="mb-2 font-semibold">Eligibility: {result.eligibility}</div>
          <div className="mb-2">Recommended Limit: <b>{result.recommended_limit}</b></div>
          <div className="mb-2">Reasoning:</div>
          <ul className="mb-4 list-disc pl-6">
            {result.reasoning?.map((r: string, i: number) => <li key={i}>{r}</li>)}
          </ul>
          <div className="flex gap-4 mb-4">
            <img src={result.plots?.shap_summary} alt="SHAP Summary" className="w-32 h-24 object-contain border" />
            <img src={result.plots?.roc_auc} alt="ROC AUC" className="w-32 h-24 object-contain border" />
          </div>
          <button className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700" onClick={onContinue}>
            Continue Journey
          </button>
        </CardContent>
      </Card>
    </div>
  );
}
