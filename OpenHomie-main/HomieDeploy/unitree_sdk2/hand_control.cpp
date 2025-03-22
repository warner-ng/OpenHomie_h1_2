#include <chrono>
#include <thread>
#include <lcm/lcm-cpp.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <iostream>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>
#include "hand_action_lcmt.hpp"

const float maxTorqueLimits_left[7]=  {  1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0   }; 
const float minTorqueLimits_left[7]=  { -1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75};
const float maxTorqueLimits_right[7]= {  1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75}; 
const float minTorqueLimits_right[7]= { -1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0    }; //1 fande 2 fande 3 

// std::array<float, 7> q_left = {};
// std::array<float, 7> q_right = {};


#define MOTOR_MAX 7
#define SENSOR_MAX 9
uint8_t hand_id = 0;

typedef struct {
    uint8_t id     : 4;
    uint8_t status : 3;
    uint8_t timeout: 1;
} RIS_Mode_t;


std::string ldds_namespace = "rt/dex3/left";
std::string lsub_namespace = "rt/dex3/left/state";
unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> lhandcmd_publisher;
unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> lhandstate_subscriber;
unitree_hg::msg::dds_::HandCmd_ msg;
unitree_hg::msg::dds_::HandState_ lstate;

std::string rdds_namespace = "rt/dex3/right";
std::string rsub_namespace = "rt/dex3/right/state";
unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> rhandcmd_publisher;
unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> rhandstate_subscriber;
unitree_hg::msg::dds_::HandState_ rstate;

std::mutex stateMutex;


class HandControl {
    private:
        lcm::LCM _simpleLCM;
        std::thread _simple_LCM_thread;
        std::thread _simple_hand_thread;
        hand_action_lcmt hand_action_simple = {0};
        float q_left[7] = {0};
        float q_right[7] = {0};
        float hand_action[14] = {0};
    public:
    HandControl(int  val){

        unitree::robot::ChannelFactory::Instance()->Init(0);
        lhandcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(ldds_namespace + "/cmd"));
        lhandstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(lsub_namespace));
        lhandcmd_publisher->InitChannel();
        lstate.motor_state().resize(MOTOR_MAX);
        lstate.press_sensor_state().resize(SENSOR_MAX);
        msg.motor_cmd().resize(MOTOR_MAX);

        rhandcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(rdds_namespace + "/cmd"));
        rhandstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(rsub_namespace));
        rhandcmd_publisher->InitChannel();
        rstate.motor_state().resize(MOTOR_MAX);
        rstate.press_sensor_state().resize(SENSOR_MAX);
        _simpleLCM.subscribe("hand_action", &HandControl::handleHandLCM, this);
        _simple_LCM_thread = std::thread(&HandControl::simpleLCMThread, this);
        _simple_hand_thread = std::thread(&HandControl::simplehandThread, this);
        for (int item = 0; item < 7; item++){
            hand_action_simple.act[item] = minTorqueLimits_left[item];
            hand_action_simple.act[item+7] = maxTorqueLimits_right[item];
        }
        hand_action_simple.act[0] = 0.0;
        hand_action_simple.act[1] = maxTorqueLimits_left[1];
        hand_action_simple.act[2] = maxTorqueLimits_left[2];
        hand_action_simple.act[7] = 0.0;
        hand_action_simple.act[8] = minTorqueLimits_right[1];
        hand_action_simple.act[9] = minTorqueLimits_right[2];
        
        
    }
    void rotateMotors(bool isLeftHand) {
        const float* maxTorqueLimits = isLeftHand ? maxTorqueLimits_left : maxTorqueLimits_right;
        const float* minTorqueLimits = isLeftHand ? minTorqueLimits_left : minTorqueLimits_right;

        float* target = isLeftHand ? q_left : q_right;

        for (int i = 0; i < MOTOR_MAX; i++) {
            RIS_Mode_t ris_mode;
            ris_mode.id = i;       
            ris_mode.status = 0x01; 
            ris_mode.timeout = 0x01; 
            
            uint8_t mode = 0;
            mode |= (ris_mode.id & 0x0F);            
            mode |= (ris_mode.status & 0x07) << 4;   
            mode |= (ris_mode.timeout & 0x01) << 7;   
            msg.motor_cmd()[i].mode(mode);
            msg.motor_cmd()[i].tau(0);
            msg.motor_cmd()[i].kp(0.5);     
            msg.motor_cmd()[i].kd(0.1);    

            // float q = mid + amplitude * sin(_count / 20000.0 * M_PI); 
            msg.motor_cmd()[i].q(target[i]);
        }

        if (isLeftHand){
            lhandcmd_publisher->Write(msg);
        }
        else{
            rhandcmd_publisher->Write(msg);
        }


        usleep(100); 


    }

    void handleHandLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const hand_action_lcmt * msg){
        (void) rbuf;
        (void) chan;
        hand_action_simple = *msg;
        
    }

    void simpleLCMThread(){
        while(true){
            _simpleLCM.handle();
        }
    }

    void simplehandThread(){
        while (true) {
            for (int i = 0; i < 7; i++){
                // q_right[i] = 0.0;
                // q_left[i] = 0.0;
                q_right[i] = hand_action_simple.act[i+7];
                q_left[i] = hand_action_simple.act[i];
                // q_right[i] += 0.01;
            }
            rotateMotors(true);
            rotateMotors(false);

        }
    }
};


int main() {
    
    
    // rmsg.motor_cmd().resize(MOTOR_MAX);
    HandControl cus(0);
    while (true) usleep(20000);
    return 0;
}