
export interface IUser {
  email: string;
  password: string;
}

const defaultUser: IUser = {
  email: "restonqwer@gmail.com",
  password: "123456"
};

export { defaultUser };
