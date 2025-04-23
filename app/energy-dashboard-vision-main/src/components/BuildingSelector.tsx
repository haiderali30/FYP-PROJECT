
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Building } from "@/types/building";

const buildings: Building[] = [
  { id: "1", name: "Hospital", address: "123 Main St", totalArea: 50000, floors: 10 },
  { id: "2", name: "School", address: "456 Park Ave", totalArea: 75000, floors: 15 },
  { id: "3", name: "Office", address: "789 Oak Rd", totalArea: 45000, floors: 8 },
  { id: "4", name: "Industry", address: "321 Pine St", totalArea: 60000, floors: 12 },
  { id: "5", name: "House 1", address: "654 Elm St", totalArea: 55000, floors: 11 },
  { id: "6", name: "House 2", address: "987 Maple Ave", totalArea: 70000, floors: 14 },
];

interface BuildingSelectorProps {
  value: Building | null;
  onChange: (building: Building | null) => void;
}

const BuildingSelector = ({ value, onChange }: BuildingSelectorProps) => {
  return (
    <Select
      value={value?.id || "all"}
      onValueChange={(id) => {
        const building = buildings.find((b) => b.id === id);
        onChange(building || null);
      }}
    >
      <SelectTrigger className="w-[200px] glass-card">
        <SelectValue placeholder="Select building" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">All Buildings</SelectItem>
        {buildings.map((building) => (
          <SelectItem key={building.id} value={building.id}>
            {building.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default BuildingSelector;

