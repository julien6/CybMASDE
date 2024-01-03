import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ServerAPIService {

  // API url
  baseApiUrl = "http://localhost:5000"

  optionRequete = {
    headers: new HttpHeaders({
      "Access-Control-Allow-Origin": "*",
      "content-type": "application/json;multipart/form-data"
    })
  };

  constructor(private http: HttpClient) { }

  // Returns an observable
  upload(file: File): Observable<any> {

    // Create form data
    const formData = new FormData();

    // Store form name as "file" with file data
    formData.append("file", file, file.name);

    // Make http post request over api
    // with formData as req
    return this.http.post(this.baseApiUrl + "/" + "worldState", formData) //, this.optionRequete)
  }

  // Returns an observable
  getRequest(request: string): Observable<any> {
    return this.http.get(this.baseApiUrl + "/" + request) //, this.optionRequete)
  }

  postRequest(request: string, data: any): Observable<any> {
    return this.http.post(this.baseApiUrl + "/" + request, data) //, this.optionRequete)
  }

  deleteRequest(request: string): Observable<any> {
    return this.http.delete(this.baseApiUrl + "/" + request) //, this.optionRequete)
  }

}