import * as React from "react"
import { cn } from "../../lib/utils"

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number
}

export function Progress({ value, className, ...props }: ProgressProps) {
  return (
    <div className={cn("w-full h-2 bg-gray-200 rounded-full", className)} {...props}>
      <div
        className="h-2 rounded-full bg-green-500 transition-all"
        style={{ width: `${value}%` }}
      />
    </div>
  )
}
