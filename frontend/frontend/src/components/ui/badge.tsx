import * as React from "react"
import { cn } from "../../lib/utils"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "green" | "amber" | "red"
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const colorMap: Record<string, string> = {
    default: "bg-gray-100 text-gray-800",
    green: "bg-green-100 text-green-800",
    amber: "bg-amber-100 text-amber-800",
    red: "bg-red-100 text-red-800",
  }
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
        colorMap[variant] || colorMap.default,
        className
      )}
      {...props}
    />
  )
}
