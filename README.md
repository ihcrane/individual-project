# Individual Project

## Project Description
I retrieved my data from Kaggle: https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset 
This dataset is a large dataset about used cars. I will be creating a machine learning model to predic the price of used cars.

## Project Goals
- To discover the drivers of value of cars
- Use the drivers to develop a ML model that determines the value of cars
- Deliever a report to a technical data science team

## Questions to answer
- Does horsepower affect the price of the vehicle?
- Does mileage affect price of the vehicle?
- Do the dimensions of the vehcile affect price?
- Does whether the car is sold by a dealer affect price?

## Intial Thoughts and Hypothesis
I believe the main drivers behind the value of the cars is horsepower, mileage, and miles per gallon. The more horsepower a car has the higher the price of the car. The lower the mileage on the vehicle the higher the price of the vehicle. The higher the miles per gallon the lower the price of the car.

## Planning
- Use the code already written prior to aid in sorting and cleaning the data
- Discover main drivers
 - First identify the drivers using statistical analysis
 - Create a pandas dataframe containing all relevant drivers as columns
- Develop a model using ML to determine vehicle value based on the top drivers
 - MVP will consist of one model per driver to test out which can most accurately predict price
 - Post MVP will consist of taking most accurate and running it through multiple models
 - Goal is to achieve an RMSE better than the baseline
- Draw and record conclusions

## Data Dictionary

| Target Variable | Definition|
|-----------------|-----------|
| price | The total price of the vehicle |

| Feature  | Definition |
|----------|------------|
| back_legroom |  Legroom in the rear seat |
| body_type | Body type of the vehicle (Hatchback, Sedan, Convertible, etc.) |
| city | City where the car is listed |
| city_mpg | Fuel economy in city traffic (km/L) |
| daysonmarket | Number of days since vehicle was listed |
| cyl | Engine configuration (I4, V6, etc.) |
| displ | Engine displacement |
| dealer | Whether the seller is a dealer |
| front_legroom | Legroom for inches in the passenger seat |
| tank_size | Fuel tank's max capacity in gallons |
| fuel_type | Dominant type of fuel used in vehicle |
| accidents | Whether the vin has any accidents registered |
| hiway_mpg | Fuel economy in highway traffic (km/L) |
| horsepower | Horsepower of the vehicle |
| new | Whether the vehicle is new |
| length | Length of vehicle in inches |
| listed_date | Date the vehicle was first listed |
| model | Model of the car |
| seats | Number of seats in the vehicle |
| mileage | The odometer reading on the vehicle |
| owners | Number of owneres the vehicle has had |
| seller_rating | Rating of the seller |
| tran | Transmission type of the vehicle (Manual, Auto, etc.) |
| drive_type | The drive train of the vehicle (FWD, RWD, etc.) |
| wheelbase | Measurement of wheelbase in inches |
| width | Width of the vehicle in inches |
| year | Year the car was made |
