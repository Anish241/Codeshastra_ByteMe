import React from "react";
import Button from "../layouts/Button";
import { getService } from "../APICalls";

const Home = () => {

  const handleClick = async () => {
    await getService();
    
  }

  return (
    <div className=" min-h-screen flex flex-col justify-center lg:px-32 px-5 text-white bg-[url('assets/img/home.png')] bg-no-repeat bg-cover opacity-90">
      <div className=" w-full lg:w-4/5 space-y-5 mt-10">
        <h1 className="text-5xl font-bold leading-tight">
          Empowering Health Choices for a Vibrant Life Your Trusted..
        </h1>
        <p>
          Our services provide guidance for asthama patients to manage their health and wellness.
          With the power of Artificial Intelligence we aim to make their life easier and healthier.
        </p>

        <Button title="Click Here to use the service" 
         handleClick={handleClick}
        />
      </div>
    </div>
  );
};

export default Home;
