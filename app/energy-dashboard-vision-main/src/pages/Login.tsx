import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';
import { Card } from '@/components/ui/card';
import { User, Lock } from 'lucide-react';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await login(username, password);
      navigate('/');
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Login Failed",
        description: "Invalid username or password. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0A0F1C]">
      <Card className="max-w-md w-full p-8 space-y-6 bg-[#111827] border-0">
        <div className="space-y-2 text-center">
          <div className="w-12 h-12 bg-[#1F2937] rounded-full flex items-center justify-center mx-auto">
            <User className="w-6 h-6 text-[#3B6BF7]" />
          </div>
          <h2 className="text-2xl font-bold text-white">
            Admin Portal
          </h2>
          <p className="text-[#9CA3AF]">
            Welcome back!
          </p>
        </div>
        <form className="space-y-5" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-[#E5E7EB] mb-1.5">
                Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-[#6B7280]" />
                </div>
                <Input
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="pl-10 bg-[#1F2937] border-0 text-white placeholder-[#6B7280] focus-visible:ring-[#3B6BF7]"
                  placeholder="Enter username"
                />
              </div>
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-[#E5E7EB] mb-1.5">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-[#6B7280]" />
                </div>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 bg-[#1F2937] border-0 text-white placeholder-[#6B7280] focus-visible:ring-[#3B6BF7]"
                  placeholder="Enter password"
                />
              </div>
            </div>
          </div>

          <Button
            type="submit"
            className="w-full bg-[#3B6BF7] hover:bg-[#2952d1] text-white py-2.5"
            disabled={isLoading}
          >
            {isLoading ? 'Signing in...' : 'Login'}
          </Button>
        </form>
      </Card>
    </div>
  );
} 