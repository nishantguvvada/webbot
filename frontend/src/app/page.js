import { Toaster } from "react-hot-toast";
import { Landing } from "./components/Landing";


export default function Home() {
  return (
    <>
      <Landing/>
      <Toaster/>
    </>
  );
}
