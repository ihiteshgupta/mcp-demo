import * as React from "react"
import { cn } from "@/lib/utils"

interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg'
}

const LoadingSpinner = React.forwardRef<HTMLDivElement, LoadingSpinnerProps>(
  ({ className, size = 'md', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "animate-spin rounded-full border-2 border-current border-t-transparent",
          {
            'h-4 w-4': size === 'sm',
            'h-6 w-6': size === 'md',
            'h-8 w-8': size === 'lg',
          },
          className
        )}
        {...props}
      />
    )
  }
)
LoadingSpinner.displayName = "LoadingSpinner"

interface LoadingOverlayProps {
  isLoading: boolean
  message?: string
  children: React.ReactNode
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  message = 'Loading...',
  children
}) => {
  return (
    <div className="relative">
      {children}
      {isLoading && (
        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="flex flex-col items-center space-y-2">
            <LoadingSpinner size="lg" />
            <p className="text-sm text-muted-foreground">{message}</p>
          </div>
        </div>
      )}
    </div>
  )
}

interface LoadingButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading?: boolean
  loadingText?: string
}

const LoadingButton = React.forwardRef<HTMLButtonElement, LoadingButtonProps>(
  ({ children, isLoading = false, loadingText = 'Loading...', disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading ? (
          <div className="flex items-center space-x-2">
            <LoadingSpinner size="sm" />
            <span>{loadingText}</span>
          </div>
        ) : (
          children
        )}
      </button>
    )
  }
)
LoadingButton.displayName = "LoadingButton"

const LoadingCard: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="flex flex-col items-center space-y-4">
        <LoadingSpinner size="lg" />
        <p className="text-muted-foreground">{message}</p>
      </div>
    </div>
  )
}

const LoadingDots: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={cn("flex space-x-1", className)}>
      <div className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
    </div>
  )
}

export { 
  LoadingSpinner, 
  LoadingOverlay, 
  LoadingButton, 
  LoadingCard, 
  LoadingDots 
}