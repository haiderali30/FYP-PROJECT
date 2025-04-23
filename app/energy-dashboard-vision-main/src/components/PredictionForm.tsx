import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/components/ui/use-toast";

const formSchema = z.object({
  hoursAhead: z.coerce
    .number()
    .min(1, "Hours ahead must be positive")
    .max(168, "Maximum prediction is 7 days (168 hours) ahead"),
  outdoorTemp: z.coerce.number(),
  humidity: z.coerce
    .number()
    .min(0, "Humidity cannot be negative")
    .max(100, "Humidity cannot exceed 100%"),
  cloudCover: z.coerce
    .number()
    .min(0, "Cloud cover cannot be negative")
    .max(100, "Cloud cover cannot exceed 100%"),
  occupancy: z.coerce
    .number()
    .min(0, "Occupancy cannot be negative")
    .max(100, "Occupancy cannot exceed 100%"),
  specialEquipment: z.coerce
    .number()
    .min(0, "Special equipment power cannot be negative"),
  lighting: z.coerce
    .number()
    .min(0, "Lighting power cannot be negative"),
  hvac: z.coerce
    .number()
    .min(0, "HVAC power cannot be negative"),
  season: z.string(),
});

const PredictionForm = () => {
  const { toast } = useToast();
  const [prediction, setPrediction] = useState<number | null>(null);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      hoursAhead: 24,
      outdoorTemp: 25,
      humidity: 60,
      cloudCover: 30,
      occupancy: 100,
      specialEquipment: 50,
      lighting: 20,
      hvac: 100,
      season: "summer",
    },
  });

  const getSeasonBooleans = (season: string) => {
    return {
      Winter: season === "winter" ? 1 : 0,
      Spring: season === "spring" ? 1 : 0,
      Summer: season === "summer" ? 1 : 0,
      Fall: season === "autumn" ? 1 : 0,
    };
  };

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      const seasonBooleans = getSeasonBooleans(values.season);
      
      const payload = {
        hours_ahead: values.hoursAhead,
        user_inputs: {
          ...seasonBooleans,
          "Outdoor Temp (°C)": values.outdoorTemp,
          "Humidity (%)": values.humidity,
          "Cloud Cover (%)": values.cloudCover,
          "Occupancy": values.occupancy,
          "Special Equipment [kW]": values.specialEquipment,
          "Lighting [kW]": values.lighting,
          "HVAC [kW]": values.hvac,
        },
      };

      console.log("Sending prediction request:", payload);

      const response = await fetch("http://localhost:5001/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        mode: "cors",
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate prediction");
      }

      const data = await response.json();
      console.log("Prediction response:", data);
      
      if (data.predicted_consumption === undefined) {
        throw new Error("Invalid response format from server");
      }
      
      setPrediction(data.predicted_consumption);
      toast({
        title: "Prediction Generated",
        description: `Predicted consumption: ${data.predicted_consumption.toFixed(2)} kWh`,
      });
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate prediction. Please try again.",
      });
    }
  };

  const handleReset = () => {
    // Reset all form fields to default values
    form.reset({
      hoursAhead: 24,
      outdoorTemp: 25,
      humidity: 60,
      cloudCover: 30,
      occupancy: 100,
      specialEquipment: 50,
      lighting: 20,
      hvac: 100,
      season: "summer",
    });
    // Clear the prediction result
    setPrediction(null);
    // Reset form state
    form.clearErrors();
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <FormField
            control={form.control}
            name="hoursAhead"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Hours Ahead</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="outdoorTemp"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Outdoor Temp (°C)</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="humidity"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Humidity (%)</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="cloudCover"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Cloud Cover (%)</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="occupancy"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Occupancy</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="specialEquipment"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Special Equipment (kW)</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="lighting"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Lighting (kW)</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="hvac"
            render={({ field }) => (
              <FormItem>
                <FormLabel>HVAC (kW)</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    {...field}
                    onChange={(e) => field.onChange(e.target.valueAsNumber)}
                    className="bg-[#0A0F1C] border-0 text-white"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <FormField
          control={form.control}
          name="season"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Season</FormLabel>
              <Select onValueChange={field.onChange} defaultValue={field.value}>
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select season" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="spring">Spring</SelectItem>
                  <SelectItem value="summer">Summer</SelectItem>
                  <SelectItem value="autumn">Autumn</SelectItem>
                  <SelectItem value="winter">Winter</SelectItem>
                </SelectContent>
              </Select>
              <FormMessage />
            </FormItem>
          )}
        />
        <div className="flex gap-4">
          <Button
            type="submit"
            className="flex-1 bg-[#3B6BF7] hover:bg-[#2952d1] text-white"
          >
            Generate Prediction
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={handleReset}
            className="flex-1"
          >
            Reset Form
          </Button>
        </div>
        
        {prediction !== null && (
          <div className="mt-4 p-4 bg-primary/10 rounded-lg">
            <h4 className="font-semibold">Predicted Consumption</h4>
            <p className="text-2xl font-bold text-primary">
              {prediction.toFixed(2)} kWh
            </p>
          </div>
        )}
      </form>
    </Form>
  );
};

export default PredictionForm;
