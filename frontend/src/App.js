import React, { useRef, useState, useEffect } from "react";

import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';
import './App.css'
import 'bootstrap/dist/css/bootstrap.min.css';
import Send from './Components/Send/Send'

import Result from './Components/Result/Result'
import 'bootstrap/dist/css/bootstrap.min.css';


  

const App =()  =>{

function handleResultClick(){
         const xhr = new XMLHttpRequest()
         xhr.open('GET', '/tests')
}
    return (
      
      
        
      <Card className="text-center">
      <Card.Header>OpenAction</Card.Header>
      <Card.Body>
        <Card.Title>Action Recognition API 😀🎉 </Card.Title>
        <Card.Text>
          Start by upploading an image to get started in testing the Action Recignition model. There are 5 kinds of actions that the model can classify namely Punch, Kick ,squat stand and wave. 
        </Card.Text>
        <Button variant="primary" onClick="window.location.href='{{ url_for( 'tests' ) }}">View Results</Button>
        <div className="Result">
          <Send
           
         
          >
         
         </Send>
        
        </div>
     

       
      </Card.Body>
      <Card.Footer className="text-muted" onClick={handleResultClick}> Made with 💘 By Ricardo</Card.Footer>
    </Card>
      
      
    
  );
    
  
    
  }
  
  export default App;


