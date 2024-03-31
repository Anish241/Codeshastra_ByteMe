import React from "react";
import img from "../assets/img/about.jpg";

const About = () => {
  return (
    <div className=" min-h-screen flex flex-col lg:flex-row justify-between items-center lg:px-32 px-5 pt-24 lg:pt-16 gap-5">
      <div className=" w-full lg:w-3/4 space-y-4">
        <h1 className=" text-4xl font-semibold text-center lg:text-start">About Us</h1>
        <p className=" text-justify lg:text-start">
        Our focus is on developing advanced systems for detecting asthma pump position and providing real-time feedback to users. By employing state-of-the-art algorithms and machine learning techniques, we have created a platform capable of accurately assessing the positioning of asthma inhalers and providing actionable insights to users.
        </p>
        <p className="text-justify lg:text-start">
        Using a combination of computer vision, sensor data, and artificial intelligence, our system analyzes the user's inhaler technique in real-time. It detects subtle movements and positions, ensuring that the medication is administered effectively. Additionally, our platform provides personalized feedback and guidance to help users optimize their inhaler usage and improve asthma management.
        </p>
        <p className="text-justify lg:text-start">
        We are excited about the potential of our technology to transform asthma care, and we welcome collaboration and feedback from the community. Whether you're a healthcare provider, a researcher, or someone living with asthma, we invite you to join us on this journey toward better asthma management.
        </p>
      </div>
      <div className=" w-full lg:w-3/4">
        <img className=" rounded-lg" src={img} alt="img" />
      </div>
    </div>
  );
};

export default About;
