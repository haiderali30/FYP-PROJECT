
import asyncHandler from 'express-async-handler';
import jwt from 'jsonwebtoken';
import { Request, Response } from 'express';
import { defaultUser } from '../models/User';

// Generate JWT
const generateToken = (email: string) => {
  return jwt.sign({ email }, process.env.JWT_SECRET || 'fallbacksecret', {
    expiresIn: '30d',
  });
};

// @desc    Auth user & get token
// @route   POST /api/auth/login
// @access  Public
export const loginUser = asyncHandler(async (req: Request, res: Response) => {
  const { email, password } = req.body;

  if (email === defaultUser.email && password === defaultUser.password) {
    res.json({
      email: defaultUser.email,
      token: generateToken(defaultUser.email),
    });
  } else {
    res.status(401);
    throw new Error('Invalid email or password');
  }
});

// @desc    Register a new user
// @route   POST /api/auth/register
// @access  Public
export const registerUser = asyncHandler(async (req: Request, res: Response) => {
  res.status(400);
  throw new Error('Registration is not available');
});
