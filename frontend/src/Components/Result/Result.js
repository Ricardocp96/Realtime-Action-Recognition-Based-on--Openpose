import React, { useEffect, useState } from 'react';
import "./Results.css"
import io from 'socket.io-client';
import Button from 'react-bootstrap/Button';
import "./Results.css"
import { Box, Image } from 'grommet';
import { useNavigate } from "react-router-dom";
const Result = () =>{
  
  const history = useNavigate();

  function handleClick() {
    history.push("/home");
  }
    
    
return(


  <Button variant="primary" onClick={handleClick} >Primary</Button>


)
     
      



    


    }


export default Result;