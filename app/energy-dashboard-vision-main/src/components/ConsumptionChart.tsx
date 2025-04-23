import { DateRange } from "react-day-picker";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { format } from "date-fns";
import { Building } from "@/types/building";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

interface ConsumptionData {
  timestamp: string;
  consumption: number;
}

interface ConsumptionChartProps {
  dateRange: DateRange;
  building: Building | null;
  data: ConsumptionData[];
  isLoading: boolean;
  viewType: 'hourly' | 'daily';
  onViewTypeChange: (value: 'hourly' | 'daily') => void;
}

const ConsumptionChart = ({ 
  dateRange, 
  building, 
  data, 
  isLoading,
  viewType,
  onViewTypeChange 
}: ConsumptionChartProps) => {
  if (isLoading) {
    return (
      <div className="h-[300px] w-full flex items-center justify-center">
        <p>Loading consumption data...</p>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="h-[300px] w-full flex items-center justify-center">
        <p>No consumption data available for the selected period</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <ToggleGroup
          type="single"
          value={viewType}
          onValueChange={(value) => value && onViewTypeChange(value as 'hourly' | 'daily')}
          className="justify-end"
        >
          <ToggleGroupItem value="hourly" aria-label="Hourly view">
            Hourly
          </ToggleGroupItem>
          <ToggleGroupItem value="daily" aria-label="Daily view">
            Daily
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="consumption" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
            <XAxis
              dataKey="timestamp"
              stroke="hsl(var(--muted-foreground))"
              tick={{ fill: "hsl(var(--muted-foreground))" }}
              tickFormatter={(value) => format(new Date(value), viewType === 'hourly' ? "MMM dd, HH:mm" : "MMM dd")}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              tick={{ fill: "hsl(var(--muted-foreground))" }}
              label={{ value: "Consumption (kW)", angle: -90, position: "insideLeft", fill: "hsl(var(--muted-foreground))" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--background))",
                border: "1px solid hsl(var(--border))",
              }}
              labelFormatter={(value) => format(new Date(value), viewType === 'hourly' ? "MMM dd, yyyy HH:mm" : "MMM dd, yyyy")}
              formatter={(value: number) => [`${value.toFixed(2)} kW`, "Consumption"]}
            />
            <Area
              type="monotone"
              dataKey="consumption"
              stroke="hsl(var(--primary))"
              fillOpacity={1}
              fill="url(#consumption)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ConsumptionChart;
