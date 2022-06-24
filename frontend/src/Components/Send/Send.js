import Dropzone from '../Dropzone/Dropzone';
import React, {Component} from "react";
import "./send.css";
import Progress from '../ProgressBar/progressBar';

import { Box, Image } from 'grommet';
class Send extends Component{
  
    constructor(props) {
        super(props);
        this.state = {
          files: [],
          uploading: false,
          uploadProgress: {},
          successfullUploaded: false,
          predicted_label:false


};
this.onFilesAdded = this.onFilesAdded.bind(this);
    this.uploadFiles = this.uploadFiles.bind(this);
    this.sendRequest = this.sendRequest.bind(this);
    this.renderActions = this.renderActions.bind(this);
    this.displayresult=this.displayresult.bind(this);
  }

  onFilesAdded(files) {
    this.setState(prevState => ({
      files: prevState.files.concat(files)
    }));
  }

  async uploadFiles() {
    this.setState({ uploadProgress: {}, uploading: true });
    const promises = [];
    this.state.files.forEach(file => {
      promises.push(this.sendRequest(file));
    });
    try {
      await Promise.all(promises);

      this.setState({ successfullUploaded: true, uploading: false });
    } catch (e) {
      // Not Production ready! Do some error handling here instead...
      this.setState({ successfullUploaded: true, uploading: false });
    }
  }

  sendRequest(file) {
  
    return new Promise((resolve, reject) => {
      const req = new XMLHttpRequest();

      req.upload.addEventListener("progress", event => {
        if (event.lengthComputable) {
          const copy = { ...this.state.uploadProgress };
          copy[file.name] = {
            state: "pending",
            percentage: (event.loaded / event.total) * 100
          };
          this.setState({ uploadProgress: copy });
        }
      });

      req.upload.addEventListener("load", event => {
        const copy = { ...this.state.uploadProgress };
        copy[file.name] = { state: "done", percentage: 100 };
        this.setState({ uploadProgress: copy });
        resolve(req.response);
      });

      req.upload.addEventListener("error", event => {
        const copy = { ...this.state.uploadProgress };
        copy[file.name] = { state: "error", percentage: 0 };
        this.setState({ uploadProgress: copy });
        reject(req.response);
      });

      const formData = new FormData();
      formData.append("file", file, file.name);
      //

      req.open("POST", "/upload", true);
      req.onreadystatechange = function () {
        if (req.readyState === 4 && req.status === 200) {
            let res = JSON.parse(req.responseText)
        
            if (req.error == true) {
                //setErrorText("Something went wrong the server");
                //setErrorColor("red");
                console.log("there was an error")
            }
            else {
              console.log(res)
              //Get the results 
              
          
              this.setState({predicted_label:true})
              

               
            }
        }

    };
      req.send(formData);

    });
  }
      


  renderProgress(file) {
    const uploadProgress = this.state.uploadProgress[file.name];
    if (this.state.uploading || this.state.successfullUploaded) {
      return (
        <div className="ProgressWrapper">
          <Progress progress={uploadProgress ? uploadProgress.percentage : 0} />
          <img
            className="CheckIcon"
            alt="done"
            src="cloud_upload_black_24dp.svg"
            style={{
              opacity:
                uploadProgress && uploadProgress.state === "done" ? 0.5 : 0
            }}
          />
        </div>
      );
    }
  }







  renderActions() {
    if (this.state.successfullUploaded) {
      return (
        <button
          onClick={() =>
            this.setState({ files: [], successfullUploaded: false })
          }
        >
          Clear
        </button>
      );
    } else {
      return (
        <button
          disabled={this.state.files.length < 0 || this.state.uploading}
          onClick={this.uploadFiles}
        >
          Upload
        </button>
      );
    }
  }
displayresult(){
if(this.state.predicted_label){
  return(

    <Box height="small" width="small">
    <Image
      fit="cover"
      src={"http://localhost:5000/getresult"}
    />
  </Box>
  );
}else{

  return(

    <Box height="small" width="small">
    <Image
      fit="cover"
      src={"http://localhost:5000/getresult"}
    />
  </Box>
  )
}
 
  
      
  
}

  render() {
    return (
      <div className="Upload">
        <span className="Title">Upload an Image </span>
        <div className="Content">
          <div>
            <Dropzone
              onFilesAdded={this.onFilesAdded}
              disabled={this.state.uploading || this.state.successfullUploaded}
            />
          </div>
      
          <div className="Files">
            {this.state.files.map(file => {
              return (
                <div key={file.name} className="Row">
                  <span className="Filename">{file.name}</span>
                  {this.renderProgress(file)}
                </div>
                
              );
            
            })}
            
           <div>
       
      </div>
      
          
          </div>
     
      
          
        </div>
      
  
        <div className="Actions">{this.renderActions()}
       
        
        
        </div>
        
     
      </div>
      

    );
  }
}

export default Send;