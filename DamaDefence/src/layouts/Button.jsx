import React from "react";

const Button = ({ title,handleClick }) => {
  return (
    <div>
      <button className=" bg-brightColor text-white px-4 py-2 rounded-md hover:bg-hoverColor transition duration-300 ease-in-out" onClick={handleClick}>
        {title}
      </button>
    </div>
  );
};

export default Button;
